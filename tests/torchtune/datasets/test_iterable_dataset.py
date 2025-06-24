# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Standalone Testing Script for TuneIterableDataset APIs.

This script implements and validates the TuneIterableDataset design, focusing on
the 99% use cases that users will encounter in practice.

CORE 99% USE CASES TESTED:
1. Basic iteration: Can I iterate through data for training?
2. Checkpointing: Can I save and resume training state?
3. DataLoader integration: Does it work with PyTorch DataLoader?
4. Dataset interleaving: Can I combine multiple datasets?
5. Error handling: Does it gracefully handle transform failures?

CURRENT ISSUE INVESTIGATION:
The tests are failing with a checkpoint resumption problem in distributed settings:
- Original run: processes batches like [[1, 2, 6, 7, 8]]  
- Resumed run: yields empty batches []

FAILED APPROACHES TRIED:
1. Pre-check iterator exhaustion: Still empty resumed batches
2. Force fresh dataset copy per epoch: Still empty resumed batches  
3. Fallback skip mechanism for empty epochs: Still empty resumed batches

CURRENT HYPOTHESIS:
The issue is NOT about exhausted iterators within epochs. The problem appears to be:
- HuggingFace dataset state restoration is not working properly in distributed/sharded scenarios
- The checkpoint save/load timing doesn't align with HF dataset's internal state updates
- When resuming, the rank's data shard becomes empty or incorrectly positioned

This suggests the problem is in the interaction between:
- StatefulDataLoader checkpointing
- HuggingFace IterableDataset state management  
- Distributed dataset sharding (split_dataset_by_node)

INSIGHTS:
An important edge case was discovered through this test script regarding
checkpointing at exact epoch boundaries.

The Problem:
When a checkpoint is saved at the exact moment a rank has finished iterating
through its data partition, the restored dataset can fail.

Example Scenario:
- Total Samples: 20
- World Size: 2 (Rank 0 and Rank 1, each gets 10 samples)
- `num_steps_before_checkpoint`: 10

Each rank processes exactly 10 samples and then saves a checkpoint. The
Hugging Face dataset's internal state dict looks like this:

'hf_dataset_state': {
    'examples_iterable': { ...
        'previous_state_example_idx': 10,  # <-- Iterator is at the end
    ... },
    'epoch': 0
}

When a new dataset loads this state, its iterator is already exhausted for
epoch 0. The `__iter__` loop runs, yields 0 samples, and incorrectly
triggers the following error:

`ValueError: Rank 0 - checkpoint_test: No samples in epoch 0. This rank's data partition may be empty.`

The fix is to allow the iterator to yield zero samples for one epoch and only
raise an error if it happens twice in a row, which would indicate a genuinely
empty data partition.

To run tests:
    $ python -m pytest planning/debug_iterable_dataset.py

The script uses realistic transforms and HuggingFace datasets to ensure
real-world applicability.
"""

import ast
import collections
import logging
import math
import tempfile
from abc import ABC, abstractmethod
from collections import Counter
from dataclasses import dataclass
from datetime import timedelta
from enum import Enum
from functools import partial
from itertools import islice
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    Iterator,
    List,
    Mapping,
    Optional,
    Protocol,
    Tuple,
    Union,
)
from unittest.mock import patch

import pyarrow as pa

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

# Note: wrap with a try-except block to avoid breaking OSS builds
from datasets import Dataset as HFDataset, load_dataset
from datasets.distributed import split_dataset_by_node

# Torchtune testing utilities
from tests.test_utils import assert_expected, gpu_test

from torch.nn.utils.rnn import pad_sequence
from torch.testing._internal.common_fsdp import FSDPTest
from torch.utils.data import DataLoader, IterableDataset
from torchdata.stateful_dataloader import StatefulDataLoader


# Configure logging
logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------------
# Class Definitions from `solving_metrics_final.md`
# --------------------------------------------------------------------------------


# An Enum to define the "how-to" for aggregation.
class AggregationType(Enum):
    """Defines how a metric's value should be aggregated."""

    SUM = "sum"
    MEAN = "mean"
    DISTRIBUTION = "distribution"
    CATEGORICAL_COUNT = "categorical_count"
    MAX = "max"
    MIN = "min"


@dataclass(frozen=True)
class Metric:
    """A self-describing metric object."""

    dataset_name: str
    name: str
    value: Union[int, float, str]
    agg_type: AggregationType


class MetricsAggregator:
    """
    Aggregates metrics across datasets and distributed ranks.

    The internal state `_state` is a dictionary where the key is a tuple
    of `(dataset_name, metric_name)` and the value is another dictionary
    holding the metric's specific state (e.g., `{'type': AggregationType.SUM, 'value': 10}`).

    Usage:
        aggregator = MetricsAggregator()
        aggregator.update(metrics)
        # Get logger-ready metrics {key: value}
        metrics = aggregator.get_metrics_for_logging(prefix="train")  # {"train/dataset1/tokens": 1234, ...}
    """

    def __init__(self, dist_window_size: int = 1000):
        # State shape: {(dataset_name, metric_name): {type: AggType, value/sum/counts/etc}}
        self._state: Dict[Tuple[str, str], Dict[str, Any]] = {}

        # For distributions, we keep a window of values to compute percentiles
        self._dist_window_size = dist_window_size

    def update(self, metrics: List[Metric]) -> None:
        """Update internal state with new metrics.

        Args:
            metrics: List of Metric objects
        """
        for metric in metrics:
            key = (metric.dataset_name, metric.name)

            if key not in self._state:
                self._initialize_state(key, metric.agg_type)

            state = self._state[key]

            # Update based on aggregation type
            if metric.agg_type == AggregationType.SUM:
                state["value"] += metric.value
            elif metric.agg_type == AggregationType.MAX:
                if state["value"] is not None:
                    state["value"] = max(state["value"], metric.value)
                else:
                    state["value"] = metric.value
            elif metric.agg_type == AggregationType.MIN:
                if state["value"] is not None:
                    state["value"] = min(state["value"], metric.value)
                else:
                    state["value"] = metric.value
            elif metric.agg_type == AggregationType.MEAN:
                state["sum"] += metric.value
                state["count"] += 1
            elif metric.agg_type == AggregationType.DISTRIBUTION:
                state["values"].append(metric.value)
            elif metric.agg_type == AggregationType.CATEGORICAL_COUNT:
                state["counts"][metric.value] += 1

    def _initialize_state(
        self, key: Tuple[str, str], agg_type: AggregationType
    ) -> None:
        """Initialize state for a new metric."""
        self._state[key] = {"type": agg_type}
        state = self._state[key]

        if agg_type == AggregationType.SUM:
            state["value"] = 0.0
        elif agg_type in (AggregationType.MAX, AggregationType.MIN):
            state["value"] = None
        elif agg_type == AggregationType.MEAN:
            state["sum"] = 0.0
            state["count"] = 0
        elif agg_type == AggregationType.DISTRIBUTION:
            state["values"] = collections.deque(maxlen=self._dist_window_size)
        elif agg_type == AggregationType.CATEGORICAL_COUNT:
            state["counts"] = collections.Counter()

    def get_metrics_for_logging(self, prefix: str = "") -> Dict[str, float]:
        """
        Returns aggregated metrics ready for logging to wandb/tensorboard.

        Args:
            prefix: Optional prefix like "train" or "valid" for metric keys

        Returns:
            Flat dictionary with keys like "train/dataset1/tokens_seen" -> float value
            Ready to be logged directly: wandb.log(metrics)
        """
        # Always compute local metrics first
        local_metrics = self._compute_local_metrics()

        # In distributed mode, perform reduction
        if dist.is_initialized() and dist.get_world_size() > 1:
            metrics = self._compute_distributed_metrics(local_metrics)
        else:
            metrics = local_metrics

        # Format for logging with proper key structure
        return self._format_for_logging(metrics, prefix)

    def _compute_local_metrics(self) -> Dict[Tuple[str, str], Dict[str, Any]]:
        """
        Compute metrics from current state.

        For distributions and categoricals, expands into multiple entries.
        The dict format allows future extensions with additional fields.

        Args:
            None

        Returns:
            Dictionary mapping (dataset_name, metric_name) -> {"value": value, "agg_type": aggregation_type}
        """
        metrics = {}

        for (ds_name, metric_name), state in self._state.items():
            agg_type = state["type"]

            if agg_type in (
                AggregationType.SUM,
                AggregationType.MAX,
                AggregationType.MIN,
            ):
                # For sum, max, and min, we just need to return the value
                metrics[(ds_name, metric_name)] = {
                    "value": state["value"],
                    "agg_type": agg_type,
                }

            elif agg_type == AggregationType.MEAN:
                if state["count"] > 0:
                    value = state["sum"] / state["count"]
                    metrics[(ds_name, metric_name)] = {
                        "value": value,
                        "agg_type": agg_type,
                    }

            elif agg_type == AggregationType.DISTRIBUTION:
                # queue -> list
                values = list(state["values"])

                # Sort to get percentiles efficiently
                sorted_values = sorted(values)
                n = len(sorted_values)

                # Each stat becomes its own metric
                # For percentiles, it is an approximattion by computing avg of averages
                metrics[(ds_name, f"{metric_name}_mean")] = {
                    "value": sum(values) / n,
                    "agg_type": AggregationType.MEAN,
                }
                metrics[(ds_name, f"{metric_name}_min")] = {
                    "value": sorted_values[0],
                    "agg_type": AggregationType.MIN,
                }
                metrics[(ds_name, f"{metric_name}_max")] = {
                    "value": sorted_values[-1],
                    "agg_type": AggregationType.MAX,
                }
                metrics[(ds_name, f"{metric_name}_p05")] = {
                    "value": sorted_values[max(0, int(0.05 * n) - 1)],
                    "agg_type": AggregationType.MEAN,
                }
                metrics[(ds_name, f"{metric_name}_p50")] = {
                    "value": sorted_values[max(0, int(0.5 * n) - 1)],
                    "agg_type": AggregationType.MEAN,
                }
                metrics[(ds_name, f"{metric_name}_p95")] = {
                    "value": sorted_values[max(0, int(0.95 * n) - 1)],
                    "agg_type": AggregationType.MEAN,
                }

            elif agg_type == AggregationType.CATEGORICAL_COUNT:
                # Expand categorical counts into individual metrics
                for category, count in state["counts"].items():
                    metrics[(ds_name, f"{metric_name}_{category}_count")] = {
                        "value": count,
                        "agg_type": AggregationType.SUM,
                    }

        return metrics

    def _compute_distributed_metrics(
        self, local_metrics: Dict[Tuple[str, str], Dict[str, Any]]
    ) -> Dict[Tuple[str, str], Dict[str, Any]]:
        """
        Performs distributed reduction on metrics.

        Strategy:
        1. Do a single all_gather_object to collect all metrics from all ranks
        2. Group metrics by key and aggregation type
        3. Apply the appropriate reduction operation locally

        This avoids complex tensor operations and handles all reduction in one pass.

        Args:
            local_metrics: Dict mapping (dataset, metric) -> {"value": value, "agg_type": agg_type, ...}

        Returns:
            Reduced metrics in same format as input

        Example:
            rank_1_metrics =
            {
                ("ds1", "metric1"): {"value": 10, "agg_type": AggregationType.SUM},
                ("ds2", "metric2"): {"value": 20, "agg_type": AggregationType.MEAN},
            }
            rank_2_metrics =
            {
                ("ds1", "metric1"): {"value": 30, "agg_type": AggregationType.SUM},
                ("ds2", "metric2"): {"value": 40, "agg_type": AggregationType.MEAN},
            }

            # After reduction
            result =
            {
                ("ds1", "metric1"): {"value": 40, "agg_type": AggregationType.SUM},
                ("ds2", "metric2"): {"value": 30, "agg_type": AggregationType.MEAN},
            }
        """
        world_size = dist.get_world_size()

        # Gather all metrics from all ranks in one operation
        dist.barrier()
        all_metrics = [None] * world_size
        dist.all_gather_object(all_metrics, local_metrics)

        # Group values by key for reduction
        grouped = collections.defaultdict(list)
        for rank_metrics in all_metrics:
            if rank_metrics:  # It's possible a rank has no metrics
                for key, metric_dict in rank_metrics.items():
                    # A key is a tuple (dataset, metric)
                    grouped[key].append(metric_dict)

        # Reduce based on aggregation type
        reduced = {}
        if not grouped:
            return reduced

        for key, metric_dicts in grouped.items():
            # All metrics for a key should have same type, just take first
            values = [m["value"] for m in metric_dicts]
            agg_type = metric_dicts[0]["agg_type"]

            # Start with copy of first dict to preserve any extra fields
            result_dict = metric_dicts[0].copy()

            if agg_type == AggregationType.SUM:
                result_dict["value"] = sum(values)
            elif agg_type == AggregationType.MAX:
                result_dict["value"] = max(values)
            elif agg_type == AggregationType.MIN:
                result_dict["value"] = min(values)
            elif agg_type == AggregationType.MEAN:
                result_dict["value"] = sum(values) / len(values)

            reduced[key] = result_dict

        return reduced

    def _format_for_logging(
        self, metrics: Dict[Tuple[str, str], Dict[str, Any]], prefix: str
    ) -> Dict[str, float]:
        """
        Format metrics for wandb/tensorboard logging.

        Args:
            metrics: Dict mapping (dataset, metric) -> {"value": value, "agg_type": agg_type, ...}
            prefix: Optional prefix like "train" or "valid"

        Returns:
            Flat dict with string keys like "train/dataset1/tokens_seen" -> float
        """
        formatted = {}

        for (ds_name, metric_name), metric_dict in metrics.items():
            # Build key: "prefix/dataset/metric" or "dataset/metric" if no prefix
            if prefix:
                key = f"{prefix}/{ds_name}/{metric_name}"
            else:
                key = f"{ds_name}/{metric_name}"

            formatted[key] = metric_dict["value"]

        return formatted

    def state_dict(self) -> Dict[str, Any]:
        """Serialize aggregator state. The state is almost directly serializable."""
        serializable_state = {}
        for key, state in self._state.items():
            state_copy = state.copy()

            # Convert non-serializable types
            if "values" in state_copy:
                state_copy["values"] = list(state_copy["values"])  # deque → list
            if "counts" in state_copy:
                state_copy["counts"] = dict(state_copy["counts"])  # Counter → dict

            # Convert tuple key to string for JSON compatibility
            # JSON doesn't support tuple keys, so we convert (dataset, metric) → "('dataset', 'metric')"
            serializable_state[str(key)] = state_copy
        return {"state": serializable_state, "dist_window_size": self._dist_window_size}

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Load aggregator state from checkpoint."""
        self._dist_window_size = state_dict["dist_window_size"]

        deserialized_state = {}
        for key_str, state in state_dict["state"].items():
            # Convert string keys back to tuples
            # "('dataset', 'metric')" → ('dataset', 'metric')
            key = ast.literal_eval(key_str)

            # Re-wrap values in their original types
            if state.get("type") == AggregationType.DISTRIBUTION:
                state["values"] = collections.deque(
                    state["values"], maxlen=self._dist_window_size
                )
            if state.get("type") == AggregationType.CATEGORICAL_COUNT:
                state["counts"] = collections.Counter(state["counts"])

            deserialized_state[key] = state
        self._state = deserialized_state


# TODO make it a Transform like other transforms
class MetricTransform(Protocol):
    """Protocol for metric transforms."""

    def set_dataset_name(self, dataset_name: str) -> None: ...
    def __call__(self, sample: Dict[str, Any]) -> Dict[str, Any]: ...


class StandardMetricTransform(MetricTransform):
    """
    Attaches per-sample metrics for tracking training progress.

    This transform is responsible for generating metrics on a per-sample
    basis (e.g., tokens per sample). The actual aggregation of these metrics
    (eg calculating sum of samples seen) is handled by the
    `MetricsAggregator`. This separation of concerns ensures that metrics are
    correctly aggregated even with multiple dataloader workers and in a
    distributed setting.

    Tracked metrics include:
    - samples_seen: A count of samples processed.
    - tokens_seen: The total number of tokens processed.
    - seq_len: A distribution of sequence lengths.
    """

    def __init__(self):
        # dataset_name is set by the dataset using set_dataset_name
        self.dataset_name: Optional[str] = None
        self.new_metric: Optional[Callable] = None

    def set_dataset_name(self, dataset_name: str) -> None:
        """Called by dataset to set the namespace for metrics.
        The dataset name is used to differentiate multiple datasets stats,
        e.g. "train/dataset1/tokens_seen" and "train/dataset2/tokens_seen"."""
        self.dataset_name = dataset_name
        self.new_metric = partial(Metric, dataset_name=dataset_name)

    def __call__(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        if self.dataset_name is None or self.new_metric is None:
            raise RuntimeError(
                "set_dataset_name() must be called before using the transform."
            )

        # Determine token key
        token_key = "tokens" if "tokens" in sample else "input_ids"
        token_len = len(sample.get(token_key, []))

        # Create metrics for this sample
        metrics = [
            self.new_metric(name="samples_seen", value=1, agg_type=AggregationType.SUM),
            self.new_metric(
                name="tokens_seen", value=token_len, agg_type=AggregationType.SUM
            ),
            self.new_metric(
                name="seq_len", value=token_len, agg_type=AggregationType.DISTRIBUTION
            ),
        ]

        # Append to existing metrics list or create new one
        if "metrics" not in sample:
            sample["metrics"] = []
        sample["metrics"].extend(metrics)
        return sample


class TuneIterableDataset(IterableDataset, ABC):
    """
    Abstract base class for all torchtune iterable datasets.
    It defines the minimal, consistent interface required for all dataset
    implementations to ensure they are compatible with the training loop,
    checkpointing, and metric logging systems.
    """

    @property
    @abstractmethod
    def dataset_name(self) -> str:
        """A unique identifier for the dataset, used for namespacing in metrics and checkpoints."""
        pass

    @abstractmethod
    def __iter__(self) -> Iterator[Dict[str, Any]]:
        """
        Returns an infinite iterator over the dataset. Each implementation is responsible
        for its own iteration logic, including shuffling and making it an infinite stream.
        """
        pass

    @abstractmethod
    def state_dict(self) -> Dict[str, Any]:
        """Returns a state dictionary for checkpointing"""
        pass

    @abstractmethod
    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Load state from a state dictionary, used when resuming from a checkpoint."""
        pass


class HFIterableDataset(TuneIterableDataset):
    """HuggingFace dataset implementation with composable metrics.

    This is an infinite dataset. After exhausting the dataset, it will restart from the beginning.

    This dataset is responsible for:
      - Loading and sharding the dataset
      - Shuffling at initialization and after each epoch
      - Applying transforms
      - Returning an infinite iterator over the dataset

      Args:
        message_transform (Optional[Callable]): Transforms raw data into Message
        model_transform (Optional[Callable]): Take messages and prepares it for the model. Usually the tokenizer.
        output_transform (Optional[Callable]): Takes tokenized inputs and prepares it for the recipe. Usually
            does some label manipulation, e.g. ignore index. Think of it as recipe-dependent, e.g. SFT, RL, DPO, etc.
        metric_transform (Optional[Callable]): Takes the sample and computes metrics, e.g. token count.
            If None, a default transform is used. To stop tracking metrics, set it to lambda x: x.
        shuffle_buffer_size (Optional[int]): Size of the shuffle buffer. If None or 0, no shuffling is done.
        seed (int): Seed for shuffling.
        num_shards_per_rank (int): Target number of shards per worker (GPU). It will find a multiple
            of world_size * dataloader_workers.
        dataset_name (Optional[str]): Name of the dataset. If None, a default name is generated
            from the path, source, and split.
        filter_fn (Optional[Callable]): Filter function to apply to the dataset.
        filter_kwargs (Optional[Dict[str, Any]]): Keyword arguments to pass to the filter function.
        load_dataset_kwargs (Dict[str, Any]): Keyword arguments to pass to the load_dataset function.

    """

    def __init__(
        self,
        *,
        message_transform: Optional[Callable] = None,
        model_transform: Optional[Callable] = None,
        output_transform: Optional[Callable] = None,
        metric_transform: Optional[MetricTransform] = None,
        shuffle_buffer_size: Optional[int] = 1000,
        seed: int = 42,
        num_shards_per_rank: int = 64,
        dataset_name: Optional[str] = None,
        filter_fn: Optional[Callable] = None,
        filter_kwargs: Optional[Dict[str, Any]] = None,
        **load_dataset_kwargs,
    ):
        # Store configuration
        self._shuffle_buffer_size = shuffle_buffer_size
        self._seed = seed
        self._message_transform = message_transform
        self._model_transform = model_transform
        self._output_transform = output_transform

        # Create default transform if not provided
        self._metric_transform = metric_transform or StandardMetricTransform()

        # Auto-generate dataset name if not provided, ensuring it's always a string.
        if dataset_name is None:
            path = load_dataset_kwargs.get("path", None)
            source = load_dataset_kwargs.get("source", None)
            split = load_dataset_kwargs.get("split", None)
            name_parts = []
            for item in [path, source, split]:
                if item is not None:
                    name_parts.append(str(item).replace("/", "_"))
            self._dataset_name: str = "_".join(name_parts)
        else:
            self._dataset_name: str = dataset_name

        # Set dataset name on the transform if it supports it
        if hasattr(self._metric_transform, "set_dataset_name"):
            self._metric_transform.set_dataset_name(self._dataset_name)

        # Internal state for resumption
        self._num_epochs = 0

        # Load and setup HF dataset
        self._setup_hf_dataset(
            load_dataset_kwargs, num_shards_per_rank, filter_fn, filter_kwargs
        )

    @property
    def dataset_name(self) -> str:
        return self._dataset_name

    def _apply_transforms(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """Apply transforms if they exist, otherwise return sample unchanged."""
        if self._message_transform is not None:
            sample = self._message_transform(sample)
        if self._model_transform is not None:
            sample = self._model_transform(sample)
        if self._output_transform is not None:
            sample = self._output_transform(sample)
        if self._metric_transform is not None:
            sample = self._metric_transform(sample)
        return sample

    def _setup_hf_dataset(
        self,
        load_dataset_kwargs: Dict[str, Any],
        num_shards_per_rank: int,
        filter_fn: Optional[Callable] = None,
        filter_kwargs: Optional[Dict[str, Any]] = None,
    ):
        """
        Configures the Hugging Face dataset, including sharding, filtering, and
        transform mapping. This method is called only once during initialization
        to avoid expensive re-computation on each epoch.
        """

        # Distributed setup
        world_size, rank = 1, 0
        if dist.is_initialized():
            world_size = dist.get_world_size()
            rank = dist.get_rank()

        # Load and shard dataset
        ds = load_dataset(**load_dataset_kwargs)

        # Use to_iterable_dataset for streaming datasets
        if not load_dataset_kwargs.get("streaming", False):

            # Define number of shards based on (world_size, num of shards per GPU, dataloader workers)
            # E.g. world_size=2, num_shards_per_rank=16, dataloader_workers=3
            # we will try 2*16 = 32 shards. Since 32 is not a multiple of 3, we will do 36 shards.
            # Each rank gets 16 shards, each dataloader worker in that rankgets 6 shards.
            worker_info = torch.utils.data.get_worker_info()
            num_dataloader_workers = worker_info.num_workers if worker_info else 1

            # Calculate total workers
            total_workers = world_size * num_dataloader_workers

            # Calculate desired shards
            desired_shards = world_size * num_shards_per_rank

            # Find the smallest multiple of total_workers that is >= desired_shards
            if desired_shards % total_workers == 0:
                num_shards = desired_shards
            else:
                num_shards = total_workers * (
                    (desired_shards + total_workers - 1) // total_workers
                )

            # If the dataset is not streaming and has a defined length,
            # we cannot have num_shards > dataset_size.
            if not load_dataset_kwargs.get("streaming", False) and hasattr(
                ds, "__len__"
            ):
                dataset_size = len(ds)
                if num_shards > dataset_size:
                    raise ValueError(
                        f"Number of shards ({num_shards}) is greater than the dataset size ({dataset_size})."
                        f"Please decrease num_shards_per_rank."
                    )

            ds = ds.to_iterable_dataset(num_shards=num_shards)

        # Shuffle the dataset
        if self._shuffle_buffer_size and self._shuffle_buffer_size > 0:
            ds = ds.shuffle(seed=self._seed, buffer_size=self._shuffle_buffer_size)

        # Distribute across ranks
        if world_size > 1:
            ds = split_dataset_by_node(ds, rank=rank, world_size=world_size)

        # Apply filtering if specified
        if filter_fn:
            filter_kwargs = filter_kwargs or {}
            ds = ds.filter(filter_fn, **filter_kwargs)

        self._ds = ds

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        """Iterate through the dataset infinitely.

        It will restart from the beginning after exhausting the dataset.

        If shuffle_buffer_size is set, it will shuffle the dataset at the beginning of each epoch
        when set_epoch is called.

        An additional metric "num_epochs" is added to the sample.
        """
        epoch_ds = self._ds

        while True:  # Infinite iteration
            epoch_seed = self._seed + self._num_epochs
            epoch_ds.set_epoch(epoch_seed)
            epoch_iterator = iter(epoch_ds)
            samples_yielded = 0

            try:
                for sample in epoch_iterator:
                    # NOTE: We apply transforms here instead of using .map() call
                    # to work around https://github.com/huggingface/datasets/issues/7630
                    # where .map() can cause incorrect resumption from a checkpoint.
                    sample = self._apply_transforms(sample)

                    # Track the number of epochs completed for each dataset. This is
                    # especially useful when interleaving multiple datasets, but
                    # also necessary to track dataset-level metrics.
                    metric_num_epochs = Metric(
                        dataset_name=self.dataset_name,
                        name="num_epochs",
                        value=self._num_epochs,
                        agg_type=AggregationType.MAX,
                    )
                    if "metrics" not in sample:
                        sample["metrics"] = []
                    sample["metrics"].append(metric_num_epochs)

                    samples_yielded += 1
                    yield sample

            except StopIteration:
                pass  # Iterator is exhausted, which is expected.
            except Exception as e:
                logger.warning(
                    f"Dataset {self.dataset_name} encountered an unexpected error: {e}."
                )
                raise

            # Check if we got zero samples - this might indicate an issue
            if samples_yielded == 0:
                logger.warning(
                    f"Dataset {self.dataset_name} epoch {self._num_epochs} yielded 0 samples - potential issue!"
                )

            # Epoch complete - increment and continue infinite loop
            self._num_epochs += 1

            # Reset to the base dataset for the next epoch's shuffling.
            epoch_ds = self._ds

    def state_dict(self) -> Dict[str, Any]:
        """
        The dataset returns its own state directly, without namespacing.
        """
        hf_state = self._ds.state_dict()
        state = {
            "num_epochs": self._num_epochs,
            "seed": self._seed,
            "hf_dataset_state": hf_state,
        }
        return state

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """
        Load state from checkpoint, including restoring the state of the
        Hugging Face IterableDataset.
        """
        self._num_epochs = state_dict["num_epochs"]
        hf_state = state_dict["hf_dataset_state"]

        # HF is responsible for resuming the dataset state
        # where it last left off
        self._ds.load_state_dict(hf_state)


class InterleavedDataset(TuneIterableDataset):
    """Infinitely interleaves multiple TuneIterableDatasets according to a list of weights.
    - The weights are normalized to sum to 1.0.
    - This dataset is responsible for managing the state of its child datasets
    to ensure correct checkpointing and resumption.

    Args:
        datasets (List[TuneIterableDataset]): List of TuneIterableDatasets to interleave.
        weights (List[float]): List of weights for each dataset. Must sum to 1.0.
        seed (int): Seed for sampling.
        dataset_name (str): Name of the dataset. If None, defaults to "interleaved_dataset".
    """

    def __init__(
        self,
        datasets: List[TuneIterableDataset],
        weights: List[float],
        seed: int,
        dataset_name: str = "interleaved_dataset",
    ):
        self._dataset_name = dataset_name

        # Preserve original order for weighted sampling
        self._dataset_names = [ds.dataset_name for ds in datasets]

        # Create a name-to-dataset mapping for robust state management
        self._datasets: Dict[str, TuneIterableDataset] = {
            ds.dataset_name: ds for ds in datasets
        }

        # Validate unique dataset names upfront - fail fast with clear error
        names = self._dataset_names
        if len(names) != len(set(names)):
            duplicates = [
                name for name, count in collections.Counter(names).items() if count > 1
            ]
            raise ValueError(
                f"Duplicate dataset names detected: {duplicates}. All {names=}"
                f"Please provide a unique 'dataset_name' for each dataset in the interleaved list."
            )

        self._sampling_generator = torch.Generator().manual_seed(seed)

        # Normalize weights to sum to 1
        total_weight = sum(weights)
        self._weights = torch.tensor(
            [w / total_weight for w in weights], dtype=torch.float
        )
        if not math.isclose(total_weight, 1.0, rel_tol=1e-9):
            logger.warning(
                f"Interleaved dataset normalized weights to sum to 1.0. Found {total_weight=}. Previous {weights=}, new {self._weights.tolist()}"
            )

    @property
    def dataset_name(self) -> str:
        return self._dataset_name

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        """Interleave samples from child infinite datasets"""
        child_iters = {name: iter(ds) for name, ds in self._datasets.items()}

        while True:
            # Sample which dataset to use
            ds_idx = torch.multinomial(
                self._weights, 1, replacement=True, generator=self._sampling_generator
            ).item()

            # Sample an index, then get the name for safe lookup
            ds_name = self._dataset_names[ds_idx]

            try:
                sample = next(child_iters[ds_name])
                yield sample
            except StopIteration:
                # Per the design, child datasets must be infinite.
                # We re-initialize to allow for continuous operation but warn loudly
                # as this may indicate a design problem in the child dataset.
                logger.warning(
                    f"Child dataset {self._datasets[ds_name].dataset_name} was exhausted. "
                    "This is unexpected for an infinite dataset. Re-initializing its iterator."
                )
                child_iters[ds_name] = iter(self._datasets[ds_name])
                sample = next(child_iters[ds_name])
                yield sample

    def state_dict(self) -> Dict[str, Any]:
        """Save state for the interleaver and its children."""
        # The parent is responsible for namespacing the child states.
        child_states = {name: ds.state_dict() for name, ds in self._datasets.items()}
        return {
            "sampling_generator_state": self._sampling_generator.get_state(),
            "child_states": child_states,
        }

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Load state for the interleaver and its children."""
        self._sampling_generator.set_state(state_dict["sampling_generator_state"])
        child_states = state_dict["child_states"]
        for name, ds in self._datasets.items():
            if name in child_states:
                # Pass the raw state dict to the child
                ds.load_state_dict(child_states[name])


# --------------------------------------------------------------------------------
# Testing
# --------------------------------------------------------------------------------

# Configure logging
logger = logging.getLogger(__name__)

# Test Constants - Avoid perfect divisions
SMALL_DATASET_SIZE = 23
MEDIUM_DATASET_SIZE = 35
SEED = 42
BATCH_SIZE = 5
DEFAULT_SHUFFLE_BUFFER_SIZE = 35

# --------------------------------------------------------------------------------
# Test Data Creation Helpers
# --------------------------------------------------------------------------------


def create_test_json_file(path: Path, num_samples: int, offset: int = 0) -> None:
    """Creates a dummy JSON test data file with token samples of varying lengths.

    Args:
        path (Path): The path to the file to create
        num_samples (int): The number of samples to create
        offset (int): The offset to add to the sample ID to ensure unique IDs in different datasets
    """
    with open(path, "w") as f:
        for i in range(num_samples):
            sample_id = i + offset
            # Realistic token length variation (1-3 tokens)
            token_len = (i % 3) + 1
            tokens = list(range(sample_id, sample_id + token_len))
            f.write(
                f'{{"id": {sample_id}, "tokens": {tokens}, "text": "sample_{sample_id}"}}\n'
            )


def collate_with_metrics(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Simple collate that extracts metrics and pads tokens."""
    all_metrics = []
    clean_batch = []
    for sample in batch:
        if "metrics" in sample:
            all_metrics.extend(sample.pop("metrics"))
        clean_batch.append(sample)

    if not clean_batch:
        return {"metrics": all_metrics}

    # Simple padding for tokens
    ids = torch.tensor([item["id"] for item in clean_batch])
    tokens = pad_sequence(
        [torch.tensor(item["tokens"]) for item in clean_batch],
        batch_first=True,
        padding_value=-1,  # Use -1 for padding to distinguish from valid IDs
    )
    collated = {
        "id": ids,
        "tokens": tokens,
    }

    # Add text field for non-tensor data
    if "text" in clean_batch[0]:
        collated["text"] = [item["text"] for item in clean_batch]

    collated["metrics"] = all_metrics
    return collated


# --------------------------------------------------------------------------------
# Test Helper Functions
# --------------------------------------------------------------------------------


# Test utility for checkpoint testing
def generate_ckpt(
    dataloader: DataLoader,
    aggregator: MetricsAggregator,
    steps_before_checkpoint: int,
    steps_after_checkpoint: int,
    resume_dataloader: Optional[DataLoader] = None,
    resume_aggregator: Optional[MetricsAggregator] = None,
) -> Dict[str, Any]:
    """
    Generates a checkpoint by running through data and saving checkpoint mid-stream.
    Optionally, a second dataloader and aggregator can be given to resume from ckpt
    and run steps_after_checkpoint to match the first one.

    Args:
        dataloader: The dataloader to test
        aggregator: The metrics aggregator to use
        steps_before_checkpoint: Number of steps to run before saving checkpoint
        steps_after_checkpoint: Number of steps to run after checkpoint
        resume_dataloader: Optional new dataloader to test resuming. If None, returns empty resumed_batches.
        resume_aggregator: Optional new aggregator to test resuming. If None, returns empty resumed_metrics.

    Returns dict with batches/metrics from both pre and post checkpoint runs.
    """
    iterator = iter(dataloader)

    # Collect batches before and after checkpoint
    batches = []
    checkpoint_state = None
    metrics_at_checkpoint = {}

    total_steps = steps_before_checkpoint + steps_after_checkpoint

    for idx, batch in enumerate(iterator):
        batches.append(batch)

        # Process metrics
        if "metrics" in batch:
            aggregator.update(batch.pop("metrics"))

        # Save checkpoint state after steps_before_checkpoint
        if idx == steps_before_checkpoint - 1:  # -1 because idx is 0-based
            checkpoint_state = {
                "loader": dataloader.state_dict(),
                "aggregator": aggregator.state_dict(),
            }
            metrics_at_checkpoint = aggregator.get_metrics_for_logging(prefix="train")

        # Stop after total steps
        if idx == total_steps - 1:
            break

    # Split batches
    pre_checkpoint_batches = batches[:steps_before_checkpoint]
    post_checkpoint_batches = batches[steps_before_checkpoint:]

    # Resume with new instances if provided
    resumed_batches = []
    resumed_metrics = {}

    if (
        resume_dataloader is not None
        and resume_aggregator is not None
        and checkpoint_state is not None
    ):
        # Test resuming with new instances
        resume_dataloader.load_state_dict(checkpoint_state["loader"])
        resume_aggregator.load_state_dict(checkpoint_state["aggregator"])
        resume_iterator = iter(resume_dataloader)

        # Collect only the post-checkpoint batches when resuming
        for idx, batch in enumerate(resume_iterator):
            resumed_batches.append(batch)

            # Process metrics
            if "metrics" in batch:
                resume_aggregator.update(batch.pop("metrics"))

            # Stop after steps_after_checkpoint
            if idx == steps_after_checkpoint - 1:
                break

        resumed_metrics = resume_aggregator.get_metrics_for_logging(prefix="train")

    return {
        # Original run
        "pre_checkpoint_batches": pre_checkpoint_batches,
        "post_checkpoint_batches": post_checkpoint_batches,
        "metrics_at_checkpoint": metrics_at_checkpoint,
        "final_metrics": aggregator.get_metrics_for_logging(prefix="train"),
        # Resumed run
        "resumed_batches": resumed_batches,
        "resumed_metrics": resumed_metrics,
        # Internal state for loading - only if someone needs to manually load
        "_checkpoint_state": checkpoint_state,
    }


# --------------------------------------------------------------------------------
# Pytest Fixtures
# --------------------------------------------------------------------------------


@pytest.fixture
def tmp_data_dir(tmp_path):
    """Provide temporary directory for test data files."""
    return tmp_path


@pytest.fixture
def small_dataset_file(tmp_data_dir):
    path = tmp_data_dir / "small_data.json"
    create_test_json_file(path, SMALL_DATASET_SIZE, offset=0)
    return str(path)


@pytest.fixture
def medium_dataset_file(tmp_data_dir):
    path = tmp_data_dir / "medium_data.json"
    create_test_json_file(path, MEDIUM_DATASET_SIZE, offset=100)
    return str(path)


@pytest.fixture
def dataset_factory():
    """Factory for creating HFIterableDataset instances with common defaults."""

    def _create_dataset(
        data_file: str,
        dataset_name: str = "test_dataset",
        shuffle: bool = False,
        message_transform: Optional[Callable] = None,
        **kwargs,
    ) -> HFIterableDataset:
        return HFIterableDataset(
            path="json",
            data_files=data_file,
            split="train",
            dataset_name=dataset_name,
            seed=SEED,
            shuffle_buffer_size=DEFAULT_SHUFFLE_BUFFER_SIZE if shuffle else 0,
            message_transform=message_transform or (lambda x: x),
            metric_transform=StandardMetricTransform(),
            num_shards_per_rank=4,
            **kwargs,
        )

    return _create_dataset


# --------------------------------------------------------------------------------
# Test Classes
# --------------------------------------------------------------------------------


class TestMetricsAggregator:
    """Focused tests for MetricsAggregator functionality."""

    @pytest.mark.parametrize(
        "agg_type,test_values,expected",
        [
            (AggregationType.SUM, [1, 2, 3, 4], 10),
            (AggregationType.MEAN, [10, 20, 30, 40], 25.0),
            (AggregationType.MAX, [-5, 10, 3, 15], 15),
            (AggregationType.MIN, [5, -2, 8, 1], -2),
            (
                AggregationType.CATEGORICAL_COUNT,
                ["A", "B", "A", "C", "A"],
                {"A": 3, "B": 1, "C": 1},
            ),
        ],
    )
    def test_aggregation_types(self, agg_type, test_values, expected):
        """Tests each `AggregationType` to ensure it computes the correct value."""
        aggregator = MetricsAggregator()

        metrics = [
            Metric(dataset_name="test", name="metric", value=val, agg_type=agg_type)
            for val in test_values
        ]
        aggregator.update(metrics)

        result = aggregator.get_metrics_for_logging()

        if agg_type == AggregationType.CATEGORICAL_COUNT:
            for category, count in expected.items():
                assert result[f"test/metric_{category}_count"] == count
        else:
            assert result["test/metric"] == expected

    def test_distribution_metrics(self):
        """Tests that `AggregationType.DISTRIBUTION` computes all expected statistics (mean, min, max, p50)."""
        aggregator = MetricsAggregator()
        values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

        metrics = [
            Metric("test", "dist_metric", val, AggregationType.DISTRIBUTION)
            for val in values
        ]
        aggregator.update(metrics)

        result = aggregator.get_metrics_for_logging(prefix="train")

        # Verify distribution statistics
        assert result["train/test/dist_metric_mean"] == 5.5
        assert result["train/test/dist_metric_min"] == 1
        assert result["train/test/dist_metric_max"] == 10
        assert (
            result["train/test/dist_metric_p50"] == 5
        )  # Median of 1-10 is 5 (index 4, value 5)

    def test_state_management(self):
        """Test aggregator checkpointing and restoration."""
        # Create aggregator with some state
        aggregator1 = MetricsAggregator()
        initial_metrics = [
            Metric("ds1", "counter", 10, AggregationType.SUM),
            Metric("ds1", "average", 5.0, AggregationType.MEAN),
            Metric("ds2", "categories", "X", AggregationType.CATEGORICAL_COUNT),
        ]
        aggregator1.update(initial_metrics)

        # Save state
        state = aggregator1.state_dict()

        # Create new aggregator and restore state
        aggregator2 = MetricsAggregator()
        aggregator2.load_state_dict(state)

        # Both should have identical metrics
        metrics1 = aggregator1.get_metrics_for_logging()
        metrics2 = aggregator2.get_metrics_for_logging()
        assert metrics1 == metrics2

        # Continue updating both - should remain identical
        additional_metrics = [
            Metric("ds1", "counter", 5, AggregationType.SUM),
            Metric("ds1", "average", 15.0, AggregationType.MEAN),
        ]
        aggregator1.update(additional_metrics)
        aggregator2.update(additional_metrics)

        final_metrics1 = aggregator1.get_metrics_for_logging()
        final_metrics2 = aggregator2.get_metrics_for_logging()
        assert final_metrics1 == final_metrics2

        # Verify expected values
        assert final_metrics1["ds1/counter"] == 15  # 10 + 5
        assert final_metrics1["ds1/average"] == 10.0  # (5 + 15) / 2


class TestHFIterableDataset:
    """Tests for single HuggingFace dataset functionality."""

    @pytest.mark.parametrize("num_epochs", [0.5, 1.0, 2.5])
    def test_epoch_boundaries_and_checkpointing(
        self, num_epochs, dataset_factory, small_dataset_file
    ):
        """
        Tests that for N epochs, each sample appears exactly N times (rounded down),
        the epoch metric is correct, and checkpointing works as expected.
        """

        # 1. Setup Dataloaders and Aggregators for original and resumed runs
        def create_loader_and_aggregator():
            dataset = dataset_factory(small_dataset_file, shuffle=False)
            loader = StatefulDataLoader(
                dataset, batch_size=BATCH_SIZE, collate_fn=collate_with_metrics
            )
            aggregator = MetricsAggregator()
            return loader, aggregator

        loader1, aggregator1 = create_loader_and_aggregator()
        loader2, aggregator2 = create_loader_and_aggregator()

        # 2. Calculate steps for the test run
        # Base calculation on actual samples we want to process
        desired_total_samples = int(SMALL_DATASET_SIZE * num_epochs)
        if desired_total_samples < BATCH_SIZE:
            raise ValueError("Not enough samples to test checkpointing")

        total_steps = desired_total_samples // BATCH_SIZE
        actual_samples_processed = total_steps * BATCH_SIZE

        # Ensure we always process at least 1 step before checkpointing
        if total_steps < 2:
            raise ValueError("Not enough steps to test checkpointing")

        steps_before_checkpoint = max(1, total_steps // 2)
        steps_after_checkpoint = total_steps - steps_before_checkpoint

        # 3. Generate checkpoint and resume
        result = generate_ckpt(
            loader1,
            aggregator1,
            steps_before_checkpoint=steps_before_checkpoint,
            steps_after_checkpoint=steps_after_checkpoint,
            resume_dataloader=loader2,
            resume_aggregator=aggregator2,
        )

        # 4. Verify checkpointing and resumption
        orig_post_ids = [b["id"].tolist() for b in result["post_checkpoint_batches"]]
        resumed_ids = [b["id"].tolist() for b in result["resumed_batches"]]
        assert (
            orig_post_ids == resumed_ids
        ), "Resumed batches should be identical for deterministic run"
        assert (
            result["final_metrics"] == result["resumed_metrics"]
        ), "Final metrics should match"

        # 5. Verify sample distribution and epoch metric
        all_batches = (
            result["pre_checkpoint_batches"] + result["post_checkpoint_batches"]
        )
        sample_ids = [id for batch in all_batches for id in batch["id"].tolist()]

        # Verify we got expected number of samples
        assert len(sample_ids) == actual_samples_processed

        # 5. Check that each sample ID appears the expected number of times
        # Simplified check: each sample should appear int(num_epochs) times,
        # with some samples appearing once more for partial epochs
        if actual_samples_processed > 0:
            counts = collections.Counter(sample_ids)

            # Each sample should appear floor(num_epochs) or ceil(num_epochs) times
            min_appearances = int(num_epochs)
            max_appearances = min_appearances + 1

            for sample_id, count in counts.items():
                assert min_appearances <= count <= max_appearances, (
                    f"Sample {sample_id} appeared {count} times, "
                    f"expected between {min_appearances} and {max_appearances} for {num_epochs} epochs"
                )

        # 6. Verify epoch metric from aggregator
        final_metrics = result["final_metrics"]
        dataset_name = loader1.dataset.dataset_name
        epoch_metric_key = f"train/{dataset_name}/num_epochs"
        # we subtract 1e-9 so that for 1.0 epochs, we don't get 1.0 but floor(0.9999)
        # and it matches the epoch_metric_key, which should be 0.0.
        assert final_metrics[epoch_metric_key] == (math.floor(num_epochs - 1e-9))

    def test_shuffling_behavior(self, dataset_factory, small_dataset_file):
        """Tests that shuffling changes data order between epochs but preserves the set of samples."""
        # Test unshuffled dataset
        unshuffled_ds = dataset_factory(
            small_dataset_file, dataset_name="unshuffled", shuffle=False
        )

        # Get samples from two passes through the dataset
        iter1 = iter(unshuffled_ds)
        epoch1_unshuffled = [s["id"] for s in islice(iter1, SMALL_DATASET_SIZE)]
        epoch2_unshuffled = [s["id"] for s in islice(iter1, SMALL_DATASET_SIZE)]

        # Unshuffled should have same order in both epochs
        assert epoch1_unshuffled == list(range(SMALL_DATASET_SIZE))
        assert epoch2_unshuffled == list(range(SMALL_DATASET_SIZE))

        # Test shuffled dataset
        shuffled_ds = dataset_factory(
            small_dataset_file, dataset_name="shuffled", shuffle=True
        )

        # Collect full epochs to compare
        iter2 = iter(shuffled_ds)
        all_epoch1_ids = [s["id"] for s in islice(iter2, SMALL_DATASET_SIZE)]
        all_epoch2_ids = [s["id"] for s in islice(iter2, SMALL_DATASET_SIZE)]

        # Shuffled epochs should have different order
        assert all_epoch1_ids != list(
            range(SMALL_DATASET_SIZE)
        ), "Shuffled should differ from original order"
        assert (
            all_epoch1_ids != all_epoch2_ids
        ), "Shuffled epochs should have different order"

        # But should contain the same set of IDs
        assert set(all_epoch1_ids) == set(range(SMALL_DATASET_SIZE))
        assert set(all_epoch2_ids) == set(range(SMALL_DATASET_SIZE))

    def test_default_dataset_name(self, small_dataset_file):
        """Test that dataset name is auto-generated from path when not provided."""
        # Create dataset without specifying name
        dataset = HFIterableDataset(
            path="json",
            data_files=small_dataset_file,
            split="train",
            # dataset_name not provided - should auto-generate
            seed=SEED,
            metric_transform=StandardMetricTransform(),
            num_shards_per_rank=4,
        )

        # Should generate name from path and split
        assert dataset.dataset_name == "json_train"

        # Test with a different path and split
        dataset2 = HFIterableDataset(
            path="json",
            data_files={"validation": small_dataset_file},
            split="validation",
            seed=SEED,
            shuffle_buffer_size=0,
            metric_transform=StandardMetricTransform(),
            num_shards_per_rank=4,
        )

        # Should generate name from path and split
        assert dataset2.dataset_name == "json_validation"


class TestInterleavedDataset:
    """Tests for multi-dataset interleaving functionality."""

    def test_initialization_validation(self, dataset_factory, small_dataset_file):
        """Tests that the dataset raises errors for invalid configurations, like duplicate names."""
        # Test duplicate dataset names
        ds1 = dataset_factory(small_dataset_file, dataset_name="duplicate")
        ds2 = dataset_factory(small_dataset_file, dataset_name="duplicate")

        with pytest.raises(ValueError, match="Duplicate dataset names detected"):
            InterleavedDataset(datasets=[ds1, ds2], weights=[0.5, 0.5], seed=SEED)

        # Test weight normalization (should work with warning)
        ds3 = dataset_factory(small_dataset_file, dataset_name="ds3")
        ds4 = dataset_factory(small_dataset_file, dataset_name="ds4")

        with patch("logging.Logger.warning") as mock_warning:
            interleaved = InterleavedDataset(
                datasets=[ds3, ds4], weights=[0.3, 0.5], seed=SEED  # Sum = 0.8, not 1.0
            )
            # Check that weights were normalized
            assert torch.allclose(interleaved._weights, torch.tensor([0.375, 0.625]))
            mock_warning.assert_called_once()

    def test_sampling_ratios(
        self, dataset_factory, small_dataset_file, medium_dataset_file
    ):
        """Tests that datasets are sampled according to their assigned weights."""
        # Create two datasets with distinct ID ranges
        # ds1 has IDs 0-22 (small dataset)
        # ds2 has IDs 100-134 (medium dataset with offset)
        ds1 = dataset_factory(small_dataset_file, dataset_name="ds1")
        ds2 = dataset_factory(medium_dataset_file, dataset_name="ds2")

        # Test with 70/30 weighting
        weights = [0.7, 0.3]
        interleaved = InterleavedDataset([ds1, ds2], weights, seed=SEED)

        # Collect 300 samples
        sample_count = 300
        samples = list(islice(iter(interleaved), sample_count))

        # Count samples by checking ID ranges
        # ds1 has IDs < 100, ds2 has IDs >= 100
        ds1_count = sum(1 for s in samples if s["id"] < 100)
        ds2_count = sum(1 for s in samples if s["id"] >= 100)

        assert ds1_count + ds2_count == sample_count

        # Check ratios are approximately correct
        ds1_ratio = ds1_count / sample_count
        ds2_ratio = ds2_count / sample_count

        # Allow 10% tolerance due to randomness
        assert abs(ds1_ratio - 0.7) < 0.1, f"ds1 ratio {ds1_ratio:.2f} should be ~0.7"
        assert abs(ds2_ratio - 0.3) < 0.1, f"ds2 ratio {ds2_ratio:.2f} should be ~0.3"

    def test_metrics_aggregation(
        self, dataset_factory, small_dataset_file, medium_dataset_file
    ):
        """Tests that metrics from all child datasets are collected and aggregated."""
        ds1 = dataset_factory(small_dataset_file, dataset_name="ds1")
        ds2 = dataset_factory(medium_dataset_file, dataset_name="ds2")

        interleaved = InterleavedDataset([ds1, ds2], [0.2, 0.8], seed=SEED)
        aggregator = MetricsAggregator()

        # Process some samples
        TOTAL_SAMPLES = 200
        for sample in islice(iter(interleaved), 200):
            aggregator.update(sample["metrics"])

        metrics = aggregator.get_metrics_for_logging()

        # Should have metrics from both datasets, with flat keys
        assert "ds1/samples_seen" in metrics
        assert "ds2/samples_seen" in metrics

        # Both datasets should have contributed samples
        assert metrics["ds1/samples_seen"] > 0
        assert metrics["ds2/samples_seen"] > 0

        # Total samples should equal what we processed
        calculated_total_samples = (
            metrics["ds1/samples_seen"] + metrics["ds2/samples_seen"]
        )
        assert calculated_total_samples == TOTAL_SAMPLES

        # Test that ratio is approximately correct
        ds1_ratio = metrics["ds1/samples_seen"] / TOTAL_SAMPLES
        ds2_ratio = metrics["ds2/samples_seen"] / TOTAL_SAMPLES

        # Allow 10% tolerance due to randomness
        assert abs(ds1_ratio - 0.2) < 0.1, f"ds1 ratio {ds1_ratio:.2f} should be ~0.2"
        assert abs(ds2_ratio - 0.8) < 0.1, f"ds2 ratio {ds2_ratio:.2f} should be ~0.8"

    def test_checkpointing(
        self, dataset_factory, small_dataset_file, medium_dataset_file
    ):
        """Tests that interleaved dataset checkpointing preserves sampling state."""

        def create_interleaved():
            ds1 = dataset_factory(small_dataset_file, dataset_name="ds1")
            ds2 = dataset_factory(medium_dataset_file, dataset_name="ds2")
            return InterleavedDataset([ds1, ds2], [0.7, 0.3], seed=SEED)

        # Original run
        interleaved1 = create_interleaved()
        loader1 = StatefulDataLoader(
            interleaved1, batch_size=BATCH_SIZE, collate_fn=collate_with_metrics
        )
        aggregator1 = MetricsAggregator()

        # Resumed run
        interleaved2 = create_interleaved()
        loader2 = StatefulDataLoader(
            interleaved2, batch_size=BATCH_SIZE, collate_fn=collate_with_metrics
        )
        aggregator2 = MetricsAggregator()

        result = generate_ckpt(
            loader1,
            aggregator1,
            steps_before_checkpoint=10,
            steps_after_checkpoint=5,
            resume_dataloader=loader2,
            resume_aggregator=aggregator2,
        )

        orig_post_ids = [b["id"].tolist() for b in result["post_checkpoint_batches"]]
        resumed_ids = [b["id"].tolist() for b in result["resumed_batches"]]
        assert (
            orig_post_ids == resumed_ids
        ), "Resumed batches should be identical for deterministic run"
        assert (
            result["final_metrics"] == result["resumed_metrics"]
        ), "Final metrics should match"


class TestDistributedInterleavedDataset(FSDPTest):
    @property
    def world_size(self) -> int:
        return 2

    @gpu_test(gpu_count=2)
    def test_distributed_interleaved_checkpointing(self):
        """Test interleaved dataset checkpointing with distributed settings.

        Assertions:
        - Each rank processes non-overlapping data shards
        - Sampling ratios (70/30) are maintained across ranks
        - Checkpoint/resume produces identical batches (deterministic)
        - Metrics correctly aggregate across ranks
        """
        rank = dist.get_rank()

        # Create a shared temp directory path (only rank 0 creates it)
        # This is critical: all ranks must read from the same data files
        # to ensure proper sharding with split_dataset_by_node
        if rank == 0:
            import tempfile

            temp_dir = tempfile.mkdtemp(prefix="distributed_test_")
        else:
            temp_dir = None

        # TODO: Prob there is a more elegant way to do this
        # Broadcast the temp directory path to all ranks
        temp_dir_list = [temp_dir]
        dist.broadcast_object_list(temp_dir_list, src=0)
        temp_dir = temp_dir_list[0]
        tmp_path = Path(temp_dir)

        try:

            def create_dataset():
                # Create interleaved dataset with no shuffle for determinism
                file1 = tmp_path / "ds1.json"
                file2 = tmp_path / "ds2.json"

                # Only rank 0 creates the files
                if rank == 0:
                    create_test_json_file(file1, SMALL_DATASET_SIZE)
                    create_test_json_file(file2, MEDIUM_DATASET_SIZE, offset=100)

                # wait for 0 rank to create files
                dist.barrier()

                ds1 = HFIterableDataset(
                    path="json",
                    data_files=str(file1),
                    split="train",
                    dataset_name="ds1",
                    shuffle_buffer_size=0,  # No shuffle for determinism
                    metric_transform=StandardMetricTransform(),
                    num_shards_per_rank=4,
                )
                ds2 = HFIterableDataset(
                    path="json",
                    data_files=str(file2),
                    split="train",
                    dataset_name="ds2",
                    shuffle_buffer_size=0,  # No shuffle for determinism
                    metric_transform=StandardMetricTransform(),
                    num_shards_per_rank=4,
                )

                return InterleavedDataset([ds1, ds2], [0.7, 0.3], seed=SEED)

            def create_dataloader(dataset):
                loader = StatefulDataLoader(
                    dataset,
                    batch_size=BATCH_SIZE,
                    num_workers=0,  # Avoid multiprocessing in distributed tests
                    collate_fn=collate_with_metrics,
                )
                return loader, MetricsAggregator()

            # Process a reasonable number of steps to ensure the test is meaningful
            # With interleaved datasets, we have plenty of data (23 + 35 = 58 samples per rank)
            steps_before = 3
            steps_after = 3
            num_samples = (steps_before + steps_after) * BATCH_SIZE
            num_samples_per_rank = num_samples // dist.get_world_size()
            assert num_samples_per_rank > 0, "Not enough samples per rank"

            # Run with checkpoint and resume
            loader1, aggregator1 = create_dataloader(create_dataset())
            loader2, aggregator2 = create_dataloader(create_dataset())

            result = generate_ckpt(
                loader1,
                aggregator1,
                steps_before,
                steps_after,
                resume_dataloader=loader2,
                resume_aggregator=aggregator2,
            )

            # 1: Deterministic resumption
            # Verify that resumed batches are identical to original post-checkpoint batches
            orig_post_ids = [
                b["id"].tolist() for b in result["post_checkpoint_batches"]
            ]
            resumed_ids = [b["id"].tolist() for b in result["resumed_batches"]]
            assert orig_post_ids == resumed_ids, (
                f"Rank {rank}: Resumed batches must be identical to original. "
                f"This indicates checkpoint/resume is not deterministic."
            )

            # 2: No duplicate samples across checkpoint boundary
            all_original_ids = []
            for batch in (
                result["pre_checkpoint_batches"] + result["post_checkpoint_batches"]
            ):
                all_original_ids.extend(batch["id"].tolist())

            # 2: Verify we got the expected number of samples
            expected_total_rank_samples = (steps_before + steps_after) * BATCH_SIZE
            assert len(all_original_ids) == expected_total_rank_samples, (
                f"Rank {rank}: Expected {expected_total_rank_samples} samples, "
                f"got {len(all_original_ids)}"
            )

            # 3: Sampling ratios maintained
            # Verify 70/30 split is approximately maintained
            ds1_samples = sum(
                1 for id in all_original_ids if id < 100
            )  # ds1 has IDs < 100
            ds2_samples = sum(
                1 for id in all_original_ids if id >= 100
            )  # ds2 has IDs >= 100
            total_samples = ds1_samples + ds2_samples

            if total_samples > 0:
                ds1_ratio = ds1_samples / total_samples
                assert 0.5 < ds1_ratio < 0.9, (
                    f"Rank {rank}: Dataset sampling ratio {ds1_ratio:.2f} outside expected "
                    f"range for 70/30 split. Got {ds1_samples}/{ds2_samples} samples."
                )

            # 4: Metrics correctly aggregated across ranks
            final_metrics = result["final_metrics"]
            resumed_metrics = result["resumed_metrics"]

            # Metrics should be identical between original and resumed runs
            assert final_metrics == resumed_metrics, (
                "Metrics differ between original and resumed runs. "
                "This indicates metric state is not properly saved/restored."
            )

            # Both datasets should have contributed samples
            assert (
                final_metrics["train/ds1/samples_seen"] > 0
            ), "ds1 contributed no samples"
            assert (
                final_metrics["train/ds2/samples_seen"] > 0
            ), "ds2 contributed no samples"

            # Total samples should match what we processed (across all ranks)
            total_seen = (
                final_metrics["train/ds1/samples_seen"]
                + final_metrics["train/ds2/samples_seen"]
            )
            expected_total = (
                (steps_before + steps_after) * BATCH_SIZE * dist.get_world_size()
            )
            assert (
                total_seen == expected_total
            ), f"Total samples seen ({total_seen}) doesn't match expected ({expected_total})"

        finally:
            # Clean up the shared temp directory (only rank 0)
            if rank == 0:
                import shutil

                shutil.rmtree(temp_dir)


class TestDistributedEpochBoundaryCheckpointing(FSDPTest):
    @property
    def world_size(self) -> int:
        return 2

    @gpu_test(gpu_count=2)
    def test_distributed_epoch_boundary_checkpointing(self):
        """The test ensures proper handling of:
        - Checkpointing at 0.9, 1.0, and 2.5 epoch boundaries
        - Correct sample distribution across epochs
        - Proper state restoration after checkpointing
        """
        rank = dist.get_rank()

        # Create a shared temp directory path (only rank 0 creates it)
        temp_dir = ""
        if rank == 0:
            import tempfile

            temp_dir = tempfile.mkdtemp(prefix="epoch_boundary_test_")

        # Broadcast the temp directory path to all ranks
        temp_dir_list = [temp_dir]
        dist.broadcast_object_list(temp_dir_list, src=0)
        temp_dir = temp_dir_list[0]
        tmp_path = Path(temp_dir)

        try:
            medium_dataset_file = tmp_path / "medium_data.json"

            # Only rank 0 creates the file
            if rank == 0:
                create_test_json_file(medium_dataset_file, MEDIUM_DATASET_SIZE)

            # Ensure all ranks wait for file to be created
            dist.barrier()

            # Test multiple epoch boundaries
            for num_epochs in [0.9, 1.0, 2.5]:

                def create_loader_and_aggregator():
                    dataset = HFIterableDataset(
                        path="json",
                        data_files=str(medium_dataset_file),
                        split="train",
                        dataset_name="epoch_boundary_test",
                        seed=SEED,
                        shuffle_buffer_size=0,  # No shuffle for determinism
                        metric_transform=StandardMetricTransform(),
                        num_shards_per_rank=2,
                    )
                    loader = StatefulDataLoader(
                        dataset,
                        batch_size=BATCH_SIZE,
                        collate_fn=collate_with_metrics,
                        num_workers=0,
                    )
                    aggregator = MetricsAggregator()
                    return loader, aggregator

                loader1, aggregator1 = create_loader_and_aggregator()
                loader2, aggregator2 = create_loader_and_aggregator()

                # Calculate steps to reach exact epoch boundary
                samples_per_rank = MEDIUM_DATASET_SIZE // dist.get_world_size()
                total_samples_to_process = int(samples_per_rank * num_epochs)

                if total_samples_to_process < BATCH_SIZE:
                    raise ValueError(
                        f"Not enough samples to process for {num_epochs} epochs"
                    )

                total_steps = total_samples_to_process // BATCH_SIZE

                if num_epochs == 1.0:
                    # Checkpoint just before the epoch boundary, ensuring we have steps left for resumption
                    steps_before_checkpoint = max(1, total_steps - 1)
                else:
                    # Checkpoint mid-training
                    steps_before_checkpoint = max(1, total_steps // 2)

                steps_after_checkpoint = total_steps - steps_before_checkpoint

                if steps_after_checkpoint <= 0:
                    raise ValueError(
                        f"Not enough samples to process for {num_epochs} epochs"
                    )

                result = generate_ckpt(
                    loader1,
                    aggregator1,
                    steps_before_checkpoint=steps_before_checkpoint,
                    steps_after_checkpoint=steps_after_checkpoint,
                    resume_dataloader=loader2,
                    resume_aggregator=aggregator2,
                )

                # 1: No empty batches after resume
                assert (
                    len(result["resumed_batches"]) > 0
                ), f"No batches after resume for {num_epochs} epochs. "

                for i, batch in enumerate(result["resumed_batches"]):
                    assert len(batch["id"]) > 0, (
                        f"Empty batch {i} after resume for {num_epochs} epochs. "
                        f"HF dataset iterator may be exhausted."
                    )

                # 2: Correct epoch metric
                final_metrics = result["final_metrics"]
                epoch_metric_key = f"train/epoch_boundary_test/num_epochs"

                # -1e-9 so that if num_epochs=1.0, expected_epoch=0
                expected_epoch = math.floor(num_epochs - 1e-9)

                assert final_metrics[epoch_metric_key] == expected_epoch, (
                    f"For {num_epochs} epochs: expected epoch metric {expected_epoch}, "
                    f"got {final_metrics[epoch_metric_key]}"
                )

                # 3: Sample distribution correct
                # Simplified check: each sample should appear int(num_epochs) times,
                # with some samples appearing once more for partial epochs
                all_ids = []
                for batch in (
                    result["pre_checkpoint_batches"] + result["post_checkpoint_batches"]
                ):
                    all_ids.extend(batch["id"].tolist())

                id_counts = Counter(all_ids)

                # Each sample should appear floor(num_epochs) or ceil(num_epochs) times
                min_appearances = int(num_epochs)
                max_appearances = min_appearances + 1

                for sample_id, count in id_counts.items():
                    assert min_appearances <= count <= max_appearances, (
                        f"Rank {rank}: Sample {sample_id} appeared {count} times, "
                        f"expected between {min_appearances} and {max_appearances} for {num_epochs} epochs"
                    )

        finally:
            # Clean up the shared temp directory (only rank 0)
            if rank == 0:
                import shutil

                shutil.rmtree(temp_dir)


class TestDistributedMetricsAggregator(FSDPTest):
    """Distributed tests for MetricsAggregator using FSDPTest infrastructure."""

    @property
    def world_size(self) -> int:
        return 2

    @gpu_test(gpu_count=2)
    def test_distributed_all_aggregation_types(self):
        """
        Test that all aggregation types work correctly in distributed setting.

        Each rank contributes different values to ensure proper reduction:
        - SUM: Should add values from all ranks
        - MEAN: Should average values across ranks
        - MAX: Should take maximum across ranks
        - MIN: Should take minimum across ranks
        - DISTRIBUTION: Should combine samples and compute stats
        - CATEGORICAL_COUNT: Should sum counts per category
        """
        aggregator = MetricsAggregator()
        rank = dist.get_rank()

        # Each rank contributes different values
        base_value = (rank + 1) * 10  # rank 0: 10, rank 1: 20

        metrics = [
            # SUM: rank 0 adds 10, rank 1 adds 20 -> total 30
            Metric("test", "sum_metric", base_value, AggregationType.SUM),
            # MEAN: rank 0 has 15, rank 1 has 25 -> avg 20
            Metric("test", "mean_metric", base_value + 5, AggregationType.MEAN),
            # MAX: rank 0 has 100, rank 1 has 200 -> max 200
            Metric("test", "max_metric", base_value * 10, AggregationType.MAX),
            # MIN: rank 0 has 5, rank 1 has 10 -> min 5
            Metric("test", "min_metric", base_value // 2, AggregationType.MIN),
        ]

        # DISTRIBUTION: Each rank adds 5 values
        # rank 0: [0, 1, 2, 3, 4], rank 1: [10, 11, 12, 13, 14]
        for i in range(5):
            metrics.append(
                Metric(
                    "test", "dist_metric", rank * 10 + i, AggregationType.DISTRIBUTION
                )
            )

        # CATEGORICAL_COUNT: Different categories per rank
        # rank 0: 3 of cat_A, 2 of cat_B
        # rank 1: 1 of cat_A, 4 of cat_C
        if rank == 0:
            metrics.extend(
                [
                    Metric(
                        "test", "cat_metric", "cat_A", AggregationType.CATEGORICAL_COUNT
                    ),
                    Metric(
                        "test", "cat_metric", "cat_A", AggregationType.CATEGORICAL_COUNT
                    ),
                    Metric(
                        "test", "cat_metric", "cat_A", AggregationType.CATEGORICAL_COUNT
                    ),
                    Metric(
                        "test", "cat_metric", "cat_B", AggregationType.CATEGORICAL_COUNT
                    ),
                    Metric(
                        "test", "cat_metric", "cat_B", AggregationType.CATEGORICAL_COUNT
                    ),
                ]
            )
        else:
            metrics.extend(
                [
                    Metric(
                        "test", "cat_metric", "cat_A", AggregationType.CATEGORICAL_COUNT
                    ),
                    Metric(
                        "test", "cat_metric", "cat_C", AggregationType.CATEGORICAL_COUNT
                    ),
                    Metric(
                        "test", "cat_metric", "cat_C", AggregationType.CATEGORICAL_COUNT
                    ),
                    Metric(
                        "test", "cat_metric", "cat_C", AggregationType.CATEGORICAL_COUNT
                    ),
                    Metric(
                        "test", "cat_metric", "cat_C", AggregationType.CATEGORICAL_COUNT
                    ),
                ]
            )

        # Update aggregator
        aggregator.update(metrics)

        # Get distributed metrics
        result = aggregator.get_metrics_for_logging(prefix="train")

        # Verify SUM aggregation
        assert (
            result["train/test/sum_metric"] == 30
        ), f"Expected sum 30, got {result['train/test/sum_metric']}"

        # Verify MEAN aggregation
        assert (
            result["train/test/mean_metric"] == 20
        ), f"Expected mean 20, got {result['train/test/mean_metric']}"

        # Verify MAX aggregation
        assert (
            result["train/test/max_metric"] == 200
        ), f"Expected max 200, got {result['train/test/max_metric']}"

        # Verify MIN aggregation
        assert (
            result["train/test/min_metric"] == 5
        ), f"Expected min 5, got {result['train/test/min_metric']}"

        # Verify DISTRIBUTION metrics
        # Combined values: [0,1,2,3,4,10,11,12,13,14]
        assert "train/test/dist_metric_mean" in result
        assert "train/test/dist_metric_min" in result
        assert "train/test/dist_metric_max" in result
        assert "train/test/dist_metric_p05" in result
        assert "train/test/dist_metric_p50" in result
        assert "train/test/dist_metric_p95" in result

        # Check distribution values
        assert result["train/test/dist_metric_min"] == 0, "Min should be 0"
        assert result["train/test/dist_metric_max"] == 14, "Max should be 14"

        # Mean should be average of local means: (2 + 12) / 2 = 7
        assert (
            result["train/test/dist_metric_mean"] == 7
        ), f"Expected mean 7, got {result['train/test/dist_metric_mean']}"

        # Verify CATEGORICAL_COUNT aggregation
        # Total: cat_A: 4, cat_B: 2, cat_C: 4
        assert (
            result["train/test/cat_metric_cat_A_count"] == 4
        ), f"Expected cat_A count 4, got {result['train/test/cat_metric_cat_A_count']}"
        assert (
            result["train/test/cat_metric_cat_B_count"] == 2
        ), f"Expected cat_B count 2, got {result['train/test/cat_metric_cat_B_count']}"
        assert (
            result["train/test/cat_metric_cat_C_count"] == 4
        ), f"Expected cat_C count 4, got {result['train/test/cat_metric_cat_C_count']}"

        # Test edge case: metrics from only one rank
        aggregator_single = MetricsAggregator()
        if rank == 0:
            aggregator_single.update(
                [
                    Metric("single", "only_rank0", 42, AggregationType.SUM),
                ]
            )

        result_single = aggregator_single.get_metrics_for_logging(prefix="train")

        # Should still work even if only one rank has the metric
        assert (
            result_single["train/single/only_rank0"] == 42
        ), f"Single rank metric failed: {result_single.get('train/single/only_rank0')}"
