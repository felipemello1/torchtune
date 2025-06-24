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
import os
import socket
import sys
import tempfile
import time
import traceback

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

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
logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
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
    A simple, stateful metrics aggregator that works in both distributed
    and non-distributed environments.
    """

    def __init__(self, dist_window_size: int = 1000):
        # TODO: add note on why this shape
        self._state: Dict[Tuple[str, str], Dict[str, Any]] = {}
        self._dist_window_size = dist_window_size

    def update(self, metrics: List[Metric]) -> None:
        """Update internal state with new metrics."""
        for metric in metrics:
            key = (metric.dataset_name, metric.name)

            if key not in self._state:
                self._initialize_state(key, metric.agg_type)

            state = self._state[key]

            # Update based on aggregation type
            if metric.agg_type == AggregationType.SUM:
                state["value"] += metric.value
            elif metric.agg_type == AggregationType.MAX:
                if state["has_value"]:
                    state["value"] = max(state["value"], metric.value)
                else:
                    state["value"] = metric.value
                    state["has_value"] = True
            elif metric.agg_type == AggregationType.MIN:
                if state["has_value"]:
                    state["value"] = min(state["value"], metric.value)
                else:
                    state["value"] = metric.value
                    state["has_value"] = True
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
            state["has_value"] = False
        elif agg_type == AggregationType.MEAN:
            state["sum"] = 0.0
            state["count"] = 0
        elif agg_type == AggregationType.DISTRIBUTION:
            state["values"] = collections.deque(maxlen=self._dist_window_size)
        elif agg_type == AggregationType.CATEGORICAL_COUNT:
            state["counts"] = collections.Counter()

    def get_metrics(self) -> Dict[str, Dict[str, Any]]:
        """Compute final metrics with distributed reduction if applicable."""
        # Handle distributed case
        if dist.is_initialized() and dist.get_world_size() > 1:
            components = self._get_local_components()
            return self._reduce_and_format_distributed(components)
        else:
            return self._compute_local_metrics()

    def _compute_local_metrics(self) -> Dict[str, Dict[str, Any]]:
        """Compute local metrics from current state for non-distributed cases."""
        report = collections.defaultdict(dict)

        for (ds_name, metric_name), state in self._state.items():
            agg_type = state["type"]

            if agg_type == AggregationType.SUM:
                report[ds_name][metric_name] = state["value"]

            elif agg_type in (AggregationType.MAX, AggregationType.MIN):
                if state["has_value"]:
                    report[ds_name][metric_name] = state["value"]

            elif agg_type == AggregationType.MEAN:
                if state["count"] > 0:
                    report[ds_name][metric_name] = state["sum"] / state["count"]
                else:
                    report[ds_name][metric_name] = 0.0

            elif agg_type == AggregationType.DISTRIBUTION:
                if state["values"]:
                    values = list(state["values"])
                    sorted_values = sorted(values)
                    n = len(sorted_values)

                    report[ds_name][f"{metric_name}_mean"] = sum(values) / n
                    report[ds_name][f"{metric_name}_min"] = sorted_values[0]
                    report[ds_name][f"{metric_name}_max"] = sorted_values[-1]
                    report[ds_name][f"{metric_name}_p05"] = sorted_values[
                        max(0, int(0.05 * n) - 1)
                    ]
                    report[ds_name][f"{metric_name}_p50"] = sorted_values[
                        max(0, int(0.5 * n) - 1)
                    ]
                    report[ds_name][f"{metric_name}_p95"] = sorted_values[
                        max(0, int(0.95 * n) - 1)
                    ]

            elif agg_type == AggregationType.CATEGORICAL_COUNT:
                for category, count in state["counts"].items():
                    report[ds_name][f"{metric_name}_{category}_count"] = count

        return dict(report)

    def _get_local_components(self) -> List[Tuple]:
        """Get components for distributed reduction."""
        local_components = []
        for (ds_name, metric_name), state in self._state.items():
            key = (ds_name, metric_name)
            agg_type = state["type"]

            if agg_type == AggregationType.SUM:
                local_components.append((key, "value", "sum", state["value"]))

            elif agg_type in (AggregationType.MAX, AggregationType.MIN):
                if state["has_value"]:
                    op_type = "max" if agg_type == AggregationType.MAX else "min"
                    local_components.append((key, "value", op_type, state["value"]))

            elif agg_type == AggregationType.MEAN:
                local_components.append((key, "sum", "sum", state["sum"]))
                local_components.append((key, "count", "sum", state["count"]))

            elif agg_type == AggregationType.DISTRIBUTION:
                if state["values"]:
                    values = list(state["values"])
                    sorted_values = sorted(values)
                    n = len(sorted_values)
                    local_components.extend(
                        [
                            (key, "mean", "mean", sum(values) / n),
                            (key, "min", "min", sorted_values[0]),
                            (key, "max", "max", sorted_values[-1]),
                            (
                                key,
                                "p05",
                                "mean",
                                sorted_values[max(0, int(0.05 * n) - 1)],
                            ),
                            (
                                key,
                                "p50",
                                "mean",
                                sorted_values[max(0, int(0.5 * n) - 1)],
                            ),
                            (
                                key,
                                "p95",
                                "mean",
                                sorted_values[max(0, int(0.95 * n) - 1)],
                            ),
                        ]
                    )
        return local_components

    def _reduce_and_format_distributed(
        self, local_components: List[Tuple]
    ) -> Dict[str, Dict[str, Any]]:
        """Perform distributed reduction and format final output."""
        # Gather all components from all ranks
        all_components = [None] * dist.get_world_size()
        dist.all_gather_object(all_components, local_components)

        # Build global component list
        all_items = [item for rank_items in all_components for item in rank_items]
        unique_keys = sorted(set((item[0], item[1], item[2]) for item in all_items))

        if not unique_keys:
            return self._compute_local_metrics()

        # Create tensors for reduction by operation type
        values_by_type = {"sum": [], "max": [], "min": [], "mean": []}
        key_map = {"sum": {}, "max": {}, "min": {}, "mean": {}}

        device = "cpu"
        if (
            dist.is_initialized()
            and torch.cuda.is_available()
            and dist.get_backend() == "nccl"
        ):
            device = torch.device(dist.get_rank())

        for reduce_type in values_by_type.keys():
            type_components = [(k[0], k[1]) for k in unique_keys if k[2] == reduce_type]
            if type_components:
                key_map[reduce_type] = {k: i for i, k in enumerate(type_components)}
                values_by_type[reduce_type] = torch.zeros(
                    len(type_components), device=device
                )

        # Fill tensors with local values
        for key, field, op_type, value in local_components:
            if op_type in key_map and (key, field) in key_map[op_type]:
                idx = key_map[op_type][(key, field)]
                values_by_type[op_type][idx] = value

        # Perform async distributed reductions
        handles = []
        if len(values_by_type["sum"]) > 0:
            handles.append(
                dist.all_reduce(
                    values_by_type["sum"], op=dist.ReduceOp.SUM, async_op=True
                )
            )
        if len(values_by_type["max"]) > 0:
            handles.append(
                dist.all_reduce(
                    values_by_type["max"], op=dist.ReduceOp.MAX, async_op=True
                )
            )
        if len(values_by_type["min"]) > 0:
            handles.append(
                dist.all_reduce(
                    values_by_type["min"], op=dist.ReduceOp.MIN, async_op=True
                )
            )
        if len(values_by_type["mean"]) > 0:
            handles.append(
                dist.all_reduce(
                    values_by_type["mean"], op=dist.ReduceOp.SUM, async_op=True
                )
            )

        # Wait for all reductions to complete
        for handle in handles:
            handle.wait()

        # Apply mean division after SUM reduction
        if len(values_by_type["mean"]) > 0:
            values_by_type["mean"] /= dist.get_world_size()

        # Reconstruct final metrics from reduced values
        reduced_data = {}
        for op_type, tensor in values_by_type.items():
            for (key, field), idx in key_map[op_type].items():
                if key not in reduced_data:
                    reduced_data[key] = {}
                reduced_data[key][field] = tensor[idx].item()

        # Handle categorical counts (non-tensor reduction)
        cat_data = {}
        for (ds_name, metric_name), state in self._state.items():
            if state["type"] == AggregationType.CATEGORICAL_COUNT:
                cat_data[(ds_name, metric_name)] = dict(state["counts"])

        all_cat_data = [None] * dist.get_world_size()
        dist.all_gather_object(all_cat_data, cat_data)

        merged_cats = collections.defaultdict(collections.Counter)
        for rank_data in all_cat_data:
            for key, counts in rank_data.items():
                merged_cats[key].update(counts)

        # Format final output
        report = collections.defaultdict(dict)

        for (ds_name, metric_name), data in reduced_data.items():
            key = (ds_name, metric_name)
            agg_type = self._state[key]["type"]

            if agg_type == AggregationType.SUM:
                report[ds_name][metric_name] = data["value"]
            elif agg_type in (AggregationType.MAX, AggregationType.MIN):
                report[ds_name][metric_name] = data["value"]
            elif agg_type == AggregationType.MEAN:
                if data["count"] > 0:
                    report[ds_name][metric_name] = data["sum"] / data["count"]
                else:
                    report[ds_name][metric_name] = 0.0
            elif agg_type == AggregationType.DISTRIBUTION:
                report[ds_name][f"{metric_name}_mean"] = data["mean"]
                report[ds_name][f"{metric_name}_min"] = data["min"]
                report[ds_name][f"{metric_name}_max"] = data["max"]
                report[ds_name][f"{metric_name}_p05"] = data["p05"]
                report[ds_name][f"{metric_name}_p50"] = data["p50"]
                report[ds_name][f"{metric_name}_p95"] = data["p95"]

        # Add categorical counts
        for (ds_name, metric_name), counter in merged_cats.items():
            for category, count in counter.items():
                report[ds_name][f"{metric_name}_{category}_count"] = count

        return dict(report)

    def state_dict(self) -> Dict[str, Any]:
        """Serialize aggregator state for checkpointing."""
        serializable_state = {}
        for key, state in self._state.items():
            state_copy = dict(state)
            if "values" in state_copy:
                state_copy["values"] = list(state_copy["values"])
            if "counts" in state_copy:
                state_copy["counts"] = dict(state_copy["counts"])
            if state_copy.get("type") in (AggregationType.MAX, AggregationType.MIN):
                state_copy["has_value"] = state.get("has_value", False)
            serializable_state[key] = state_copy
        return {"state": serializable_state, "dist_window_size": self._dist_window_size}

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Load aggregator state from checkpoint."""
        self._dist_window_size = state_dict["dist_window_size"]
        self._state = {}

        for key, state in state_dict["state"].items():
            key = ast.literal_eval(key) if isinstance(key, str) else key
            self._state[key] = dict(state)
            if "values" in state:
                self._state[key]["values"] = collections.deque(
                    state["values"], maxlen=self._dist_window_size
                )
            if "counts" in state:
                self._state[key]["counts"] = collections.Counter(state["counts"])
            if self._state[key].get("type") in (
                AggregationType.MAX,
                AggregationType.MIN,
            ):
                self._state[key]["has_value"] = state.get("has_value", False)


class StandardMetricTransform:
    def __init__(self, dataset_name: str):
        self.dataset_name = dataset_name
        self.new_metric = partial(Metric, dataset_name=dataset_name)

    def __call__(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        # User-defined logic to extract metrics from a sample
        token_key = "tokens" if "tokens" in sample else "input_ids"
        token_len = len(sample.get(token_key, []))

        metrics = [
            self.new_metric(name="samples_seen", value=1, agg_type=AggregationType.SUM),
            self.new_metric(
                name="tokens_seen", value=token_len, agg_type=AggregationType.SUM
            ),
            self.new_metric(
                name="seq_len", value=token_len, agg_type=AggregationType.DISTRIBUTION
            ),
        ]
        # Use a consistent key to store the metrics list
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
        """
        Return a state dictionary for checkpointing. The state should be namespaced
        by the dataset_name to prevent key collisions when datasets are composed.

        Example format: {self.dataset_name: {state_data}}
        """
        pass

    @abstractmethod
    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Load state from a state dictionary, used when resuming from a checkpoint."""
        pass


class HFIterableDataset(TuneIterableDataset):
    """
    Hugging Face dataset implementation with composable metrics.
    Lazy operations are correctly handled during setup, not in the iterator.
    """

    def __init__(
        self,
        *,
        message_transform: Optional[Callable] = None,
        model_transform: Optional[Callable] = None,
        output_transform: Optional[Callable] = None,
        metric_transform: Optional[Callable] = None,
        shuffle_buffer_size: Optional[int] = 1000,
        seed: int = 42,
        num_shards_per_worker: int = 64,
        dataset_name: Optional[str] = None,
        filter_fn: Optional[Callable] = None,
        filter_kwargs: Optional[Dict[str, Any]] = None,
        max_transform_failures_per_epoch: int = 100,
        **load_dataset_kwargs,
    ):
        # Auto-generate dataset name if not provided
        if dataset_name is None:
            path = load_dataset_kwargs.get("path", "unknown")
            split = load_dataset_kwargs.get("split", "train")
            self._dataset_name = f"{path.replace('/', '_')}_{split}"
        else:
            self._dataset_name = dataset_name

        # Store configuration
        self._shuffle_buffer_size = shuffle_buffer_size
        self._seed = seed
        self._transforms = {
            "message": message_transform or (lambda x: x),
            "model": model_transform or (lambda x: x),
            "output": output_transform or (lambda x: x),
            "metric": metric_transform or (lambda x: x),
        }

        # Internal state for resumption
        self._num_epochs = 0

        # NOTE: every rank and dataloader MP has its own counter
        self._max_transform_failures_per_epoch = max_transform_failures_per_epoch
        self._transform_failures_this_epoch = 0

        # Load and setup HF dataset
        self._setup_hf_dataset(
            load_dataset_kwargs, num_shards_per_worker, filter_fn, filter_kwargs
        )

    @property
    def dataset_name(self) -> str:
        return self._dataset_name

    def _setup_hf_dataset(
        self,
        load_dataset_kwargs: Dict[str, Any],
        num_shards_per_worker: int,
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
            worker_info = torch.utils.data.get_worker_info()
            num_dataloader_workers = worker_info.num_workers if worker_info else 1
            total_workers = world_size * num_dataloader_workers
            desired_shards = world_size * num_shards_per_worker

            # least common multiplier
            num_shards = (desired_shards * total_workers) // math.gcd(
                desired_shards, total_workers
            )

            # Ensure there are at least as many shards as total workers
            if num_shards < total_workers:
                num_shards = total_workers

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

        # TODO: .map skips samples when resuming from checkpoint.
        # See https://github.com/huggingface/datasets/issues/7630
        # until this is resolved, we apply transforms in __iter__
        # self._ds = ds.map(self._apply_transforms).filter(self._filter_failed_transforms)
        self._ds = ds

    def _apply_transforms(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """Apply the transform pipeline"""
        sample = self._transforms["message"](sample)
        sample = self._transforms["model"](sample)
        sample = self._transforms["output"](sample)
        return sample

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        """Iterate through the dataset with automatic metrics collection."""
        epoch_ds = self._ds

        while True:  # Infinite iteration
            self._transform_failures_this_epoch = 0
            epoch_seed = self._seed + self._num_epochs

            epoch_ds.set_epoch(epoch_seed)
            epoch_iterator = iter(epoch_ds)

            for sample in epoch_iterator:
                sample = self._apply_transforms(sample)
                sample = self._transforms["metric"](sample)

                # add num_epochs metric so in multi-dataset we
                # can keep track of which epoch we are in for each dataset
                metric_num_epochs = Metric(
                    dataset_name=self.dataset_name,
                    name="num_epochs",
                    value=self._num_epochs,
                    agg_type=AggregationType.SUM,
                )
                if "metrics" not in sample:
                    sample["metrics"] = []
                sample["metrics"].append(metric_num_epochs)

                yield sample

            # Epoch complete - increment and continue infinite loop
            self._num_epochs += 1

            # Reset to the base dataset for the next epoch's shuffling.
            epoch_ds = self._ds

    def state_dict(self) -> Dict[str, Any]:
        """
        Return state for checkpointing, including the state of the underlying
        Hugging Face IterableDataset to ensure exact resumption.
        """
        hf_dataset_state = self._ds.state_dict()

        state = {
            "num_epochs": self._num_epochs,
            "seed": self._seed,
            "hf_dataset_state": hf_dataset_state,
            "transform_failures_this_epoch": self._transform_failures_this_epoch,
        }
        return {self.dataset_name: state}

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """
        Load state from checkpoint, including restoring the state of the
        Hugging Face IterableDataset.
        """
        state = state_dict[self.dataset_name]
        self._num_epochs = state["num_epochs"]

        # HF is responsible for resuming the dataset state
        # where it last left off
        self._ds.load_state_dict(state["hf_dataset_state"])

class InterleavedDataset(TuneIterableDataset):
    """
    Interleaves multiple datasets with specified weights.
    Demonstrates simple, direct state management without inheritance complexity.
    """

    def __init__(
        self,
        datasets: List[TuneIterableDataset],
        weights: List[float],
        seed: int,
        dataset_name: str = "interleaved_dataset",
    ):
        self._dataset_name = dataset_name
        self._datasets = datasets
        self._seed = seed 
        self._sampling_generator = torch.Generator().manual_seed(seed)

        # Validate unique dataset names upfront - fail fast with clear error
        names = [ds.dataset_name for ds in datasets]
        if len(names) != len(set(names)):
            duplicates = [
                name for name, count in collections.Counter(names).items() if count > 1
            ]
            raise ValueError(
                f"Duplicate dataset names detected: {duplicates}. All {names=}"
                f"Please provide a unique 'dataset_name' for each dataset in the interleaved list."
            )

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
        child_iters = [iter(ds) for ds in self._datasets]

        while True:
            # Sample which dataset to use
            ds_idx = torch.multinomial(
                self._weights, 1, replacement=True, generator=self._sampling_generator
            ).item()

            try:
                sample = next(child_iters[ds_idx])
                yield sample
            except StopIteration:
                # Per the design, child datasets must be infinite.
                # We re-initialize to allow for continuous operation but warn loudly
                # as this may indicate a design problem in the child dataset.
                logger.warning(
                    f"Child dataset {self._datasets[ds_idx].dataset_name} was exhausted. "
                    "This is unexpected for an infinite dataset. Re-initializing its iterator."
                )
                child_iters[ds_idx] = iter(self._datasets[ds_idx])
                sample = next(child_iters[ds_idx])
                yield sample

    def state_dict(self) -> Dict[str, Any]:
        """Save state for the interleaver and its children."""
        child_states = {ds.dataset_name: ds.state_dict() for ds in self._datasets}

        state = {
            "seed": self._seed,
            "sampling_generator_state": self._sampling_generator.get_state(),
            "child_states": child_states,
        }

        return {self.dataset_name: state}

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Load state for the interleaver and its children."""
        own_state = state_dict[self.dataset_name]

        # Restore seed if present (for backward compatibility)
        if "seed" in own_state:
            self._seed = own_state["seed"]

        self._sampling_generator.set_state(own_state["sampling_generator_state"])

        child_states = own_state["child_states"]
        for ds in self._datasets:
            if ds.dataset_name in child_states:
                ds.load_state_dict({ds.dataset_name: child_states[ds.dataset_name]})


# --------------------------------------------------------------------------------
# Testing
# --------------------------------------------------------------------------------

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Test Constants - Avoid perfect divisions
SMALL_DATASET_SIZE = 23
MEDIUM_DATASET_SIZE = 35
LARGE_DATASET_SIZE = 47
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
    """A collate function that extracts metrics before padding the rest of the batch."""
    all_metrics = []
    clean_batch = []

    for sample in batch:
        if "metrics" in sample:
            all_metrics.extend(sample.pop("metrics"))
        clean_batch.append(sample)

    if not clean_batch:
        return {"metrics": all_metrics}

    # Collate the actual data
    collated = {
        "id": torch.tensor([item["id"] for item in clean_batch]),
        "tokens": pad_sequence(
            [torch.tensor(item["tokens"]) for item in clean_batch], batch_first=True
        ),
    }

    # Add text field for non-tensor data
    if "text" in clean_batch[0]:
        collated["text"] = [item["text"] for item in clean_batch]

    # Generic handling for other non-tensor data
    all_keys = set(k for item in clean_batch for k in item.keys())
    for key in all_keys:
        if key not in collated and isinstance(
            clean_batch[0].get(key), (str, int, float)
        ):
            collated[key] = [item.get(key) for item in clean_batch]

    collated["metrics"] = all_metrics
    return collated


# --------------------------------------------------------------------------------
# Test Helper Functions
# --------------------------------------------------------------------------------


def run_training_loop(
    dataloader_iter: Iterator[Dict[str, Any]],
    aggregator: MetricsAggregator,
    num_steps: int,
) -> Tuple[List[Dict[str, Any]], Dict[str, Dict[str, Any]]]:
    """
    Simulate a training loop segment - the most common testing pattern.
    Returns collected batches and final metrics.
    """
    collected_batches = []

    for batch in islice(dataloader_iter, num_steps):
        # Update metrics (simulating training loop behavior)
        if "metrics" in batch:
            aggregator.update(batch.pop("metrics"))

        collected_batches.append(batch)

    return collected_batches, aggregator.get_metrics()


def assert_sample_structure(sample: Dict[str, Any], context: str = "") -> None:
    """Asserts that a sample has the required keys and data types."""
    required_keys = ["id", "tokens", "metrics"]
    for key in required_keys:
        assert (
            key in sample
        ), f"{context}: Missing required key '{key}'. Available keys: {list(sample.keys())}"

    assert isinstance(
        sample["tokens"], list
    ), f"{context}: 'tokens' should be list, got {type(sample['tokens'])}"
    assert isinstance(
        sample["metrics"], list
    ), f"{context}: 'metrics' should be list, got {type(sample['metrics'])}"
    assert len(sample["metrics"]) > 0, f"{context}: 'metrics' should not be empty"


def assert_metrics_structure(
    metrics: Dict[str, Dict[str, Any]], expected_datasets: List[str]
) -> None:
    """Asserts that the metrics dictionary has the expected structure and numerical values."""
    assert isinstance(metrics, dict), f"Metrics should be dict, got {type(metrics)}"

    for dataset_name in expected_datasets:
        assert (
            dataset_name in metrics
        ), f"Missing dataset '{dataset_name}' in metrics. Available: {list(metrics.keys())}"

        ds_metrics = metrics[dataset_name]
        assert isinstance(
            ds_metrics, dict
        ), f"Metrics for '{dataset_name}' should be dict, got {type(ds_metrics)}"

        # Verify all metric values are numeric (for logging compatibility)
        for metric_name, value in ds_metrics.items():
            assert isinstance(
                value, (int, float)
            ), f"Metric '{dataset_name}/{metric_name}' should be numeric, got {type(value)}: {value}"


def assert_checkpoint_continuation(
    original_iter: Iterator,
    restored_dataset: TuneIterableDataset,
    num_samples: int = 5,
    exact_match: bool = True,
) -> None:
    """
    Asserts that a restored dataset continues iterating from where the original left off.
    For shuffled datasets, exact_match can be set to False to check structural consistency.
    """
    # Get continuation from original iterator
    original_continuation = []
    for _ in range(num_samples):
        try:
            sample = next(original_iter)
            # Use a tuple of (id, source)
            key = (sample.get("id"), sample.get("source", "default"))
            original_continuation.append(key)
        except StopIteration:
            break

    # Get samples from restored dataset
    restored_samples = []
    for sample in islice(iter(restored_dataset), len(original_continuation)):
        key = (sample.get("id"), sample.get("source", "default"))
        restored_samples.append(key)

    if exact_match:
        assert original_continuation == restored_samples, (
            f"Restored dataset continuation mismatch.\n"
            f"Original: {original_continuation}\n"
            f"Restored: {restored_samples}"
        )
    else:
        # For shuffled data, verify structural consistency and sample diversity
        assert len(original_continuation) == len(
            restored_samples
        ), f"Sample count mismatch: {len(original_continuation)} vs {len(restored_samples)}"
        # Both should have valid sample IDs
        assert all(
            isinstance(x[0], int) for x in original_continuation
        ), "Original samples should have integer IDs"
        assert all(
            isinstance(x[0], int) for x in restored_samples
        ), "Restored samples should have integer IDs"

        # Check that we don't have all the same samples (which would indicate a reset, not a shuffle)
        if len(original_continuation) > 1:
            assert (
                len(set(original_continuation)) > 1
            ), "Original continuation lacks diversity, shuffle might not be effective"
            assert (
                len(set(restored_samples)) > 1
            ), "Restored samples lack diversity, shuffle might not be effective"


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
def large_dataset_file(tmp_data_dir):
    path = tmp_data_dir / "large_data.json"
    create_test_json_file(path, LARGE_DATASET_SIZE, offset=1000)
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
            model_transform=lambda x: x,
            output_transform=lambda x: x,
            metric_transform=StandardMetricTransform(dataset_name),
            num_shards_per_worker=4,
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

        result = aggregator.get_metrics()

        if agg_type == AggregationType.CATEGORICAL_COUNT:
            for category, count in expected.items():
                assert result["test"][f"metric_{category}_count"] == count
        else:
            assert result["test"]["metric"] == expected

    def test_distribution_metrics(self):
        """Tests that `AggregationType.DISTRIBUTION` computes all expected statistics (mean, min, max, p50)."""
        aggregator = MetricsAggregator()
        values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

        metrics = [
            Metric("test", "dist", val, AggregationType.DISTRIBUTION) for val in values
        ]
        aggregator.update(metrics)

        result = aggregator.get_metrics()

        # Verify distribution statistics
        assert result["test"]["dist_mean"] == 5.5
        assert result["test"]["dist_min"] == 1
        assert result["test"]["dist_max"] == 10
        assert result["test"]["dist_p50"] == 5  # Median of 1-10 is 5 (index 4, value 5)

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
        metrics1 = aggregator1.get_metrics()
        metrics2 = aggregator2.get_metrics()
        assert metrics1 == metrics2

        # Continue updating both - should remain identical
        additional_metrics = [
            Metric("ds1", "counter", 5, AggregationType.SUM),
            Metric("ds1", "average", 15.0, AggregationType.MEAN),
        ]
        aggregator1.update(additional_metrics)
        aggregator2.update(additional_metrics)

        final_metrics1 = aggregator1.get_metrics()
        final_metrics2 = aggregator2.get_metrics()
        assert final_metrics1 == final_metrics2

        # Verify expected values
        assert final_metrics1["ds1"]["counter"] == 15  # 10 + 5
        assert final_metrics1["ds1"]["average"] == 10.0  # (5 + 15) / 2


class TestHFIterableDataset:
    """Tests for single HuggingFace dataset functionality."""

    @pytest.mark.parametrize(
        "dataset_size,expected_epochs",
        [
            ("small", 0.5),  # < 1 epoch
            ("medium", 1.2),  # > 1 epoch
            ("large", 2.3),  # > 2 epochs
        ],
    )
    def test_epoch_boundaries(
        self, dataset_size, expected_epochs, dataset_factory, request
    ):
        """Tests that the dataset can be iterated over for more or less than a full epoch."""
        # Get the appropriate dataset file
        dataset_file = request.getfixturevalue(f"{dataset_size}_dataset_file")
        dataset = dataset_factory(dataset_file, shuffle=False)

        # Calculate samples to iterate through
        dataset_sizes = {
            "small": SMALL_DATASET_SIZE,
            "medium": MEDIUM_DATASET_SIZE,
            "large": LARGE_DATASET_SIZE,
        }
        total_samples = int(dataset_sizes[dataset_size] * expected_epochs)

        samples = list(islice(iter(dataset), total_samples))

        # Verify we got expected number of samples
        assert len(samples) == total_samples

        # For multiple epochs, verify data repeats correctly (no shuffle)
        if expected_epochs > 1:
            epoch_size = dataset_sizes[dataset_size]
            first_epoch_ids = [s["id"] for s in samples[:epoch_size]]
            second_epoch_start = epoch_size
            second_epoch_end = min(epoch_size * 2, len(samples))
            second_epoch_ids = [
                s["id"] for s in samples[second_epoch_start:second_epoch_end]
            ]

            # They should start the same way (as many samples as we have in second epoch)
            comparison_length = len(second_epoch_ids)
            assert (
                first_epoch_ids[:comparison_length] == second_epoch_ids
            ), "Epochs should repeat identical data without shuffling"

    @pytest.mark.parametrize("shuffle", [False, True])
    def test_shuffling_behavior(self, shuffle, dataset_factory, medium_dataset_file):
        """Tests that shuffling changes data order and is reproducible with the same seed."""
        dataset1 = dataset_factory(medium_dataset_file, shuffle=shuffle)
        dataset2 = dataset_factory(medium_dataset_file, shuffle=shuffle)  # Same seed

        samples1 = [s["id"] for s in islice(iter(dataset1), 20)]
        samples2 = [s["id"] for s in islice(iter(dataset2), 20)]

        if shuffle:
            # Different from non-shuffled, but reproducible with same seed
            no_shuffle_dataset = dataset_factory(medium_dataset_file, shuffle=False)
            no_shuffle_samples = [s["id"] for s in islice(iter(no_shuffle_dataset), 20)]

            assert (
                samples1 != no_shuffle_samples
            ), "Shuffled order should differ from original"
            assert samples1 == samples2, "Same seed should produce identical shuffle"
        else:
            # Should be identical and in original order
            assert samples1 == samples2
            expected_ids = list(range(20))  # IDs 0-19 for first 20 samples
            assert (
                samples1 == expected_ids
            ), "Non-shuffled should maintain original order"

    @pytest.mark.parametrize(
        "checkpoint_timing",
        ["very_early", "mid_epoch", "epoch_boundary", "multi_epoch"],
    )
    def test_checkpointing(
        self, checkpoint_timing, dataset_factory, large_dataset_file
    ):
        """Tests that dataset state can be saved and restored at various points in an epoch."""
        dataset1 = dataset_factory(large_dataset_file, shuffle=False)
        iterator1 = iter(dataset1)

        # Determine checkpoint position - test edge cases
        if checkpoint_timing == "very_early":
            consume_count = 1  # After just 1 sample
        elif checkpoint_timing == "mid_epoch":
            consume_count = LARGE_DATASET_SIZE // 3  # Not a clean fraction
        elif checkpoint_timing == "epoch_boundary":
            consume_count = LARGE_DATASET_SIZE
        else:  # multi_epoch
            consume_count = int(
                LARGE_DATASET_SIZE * 1.7
            )  # Partway through second epoch

        # Consume samples and checkpoint
        consumed = list(islice(iterator1, consume_count))
        state = dataset1.state_dict()

        # Create new dataset and restore
        dataset2 = dataset_factory(large_dataset_file, shuffle=False)
        dataset2.load_state_dict(state)

        # Verify continuation is correct
        assert_checkpoint_continuation(
            iterator1,
            dataset2,
            num_samples=min(10, LARGE_DATASET_SIZE),
            exact_match=True,  # No shuffle, so should be exact
        )


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
            mock_warning.assert_called_once()

    @pytest.mark.parametrize(
        "weights,sample_count",
        [
            ([0.5, 0.5], 300),
            ([0.9, 0.1], 300), 
        ],
    )
    def test_sampling_ratios(
        self, weights, sample_count, dataset_factory, small_dataset_file
    ):
        """Tests that datasets are sampled according to their assigned weights."""
        datasets = []
        expected_datasets = []

        for i, weight in enumerate(weights):
            # Create separate data files for each dataset
            dataset_name = f"ds_{i}"
            datasets.append(dataset_factory(small_dataset_file, dataset_name=dataset_name))
            expected_datasets.append(dataset_name)

        interleaved = InterleavedDataset(datasets, weights, seed=SEED)

        # Collect samples and count by dataset
        samples = list(islice(iter(interleaved), sample_count))
        dataset_counts = collections.Counter()

        for sample in samples:
            # Find which dataset this sample came from by checking metrics
            for metric in sample["metrics"]:
                if metric.name == "samples_seen":
                    dataset_counts[metric.dataset_name] += 1
                    break

        # Verify sampling ratios are approximately correct
        total_samples = sum(dataset_counts.values())
        assert (
            total_samples == sample_count
        ), f"Expected {sample_count} samples, got {total_samples}"

        for i, (dataset_name, expected_weight) in enumerate(
            zip(expected_datasets, weights)
        ):
            observed_ratio = dataset_counts[dataset_name] / total_samples

            tolerance = 0.15

            assert abs(observed_ratio - expected_weight) < tolerance, (
                f"Dataset {dataset_name}: expected ratio {expected_weight:.2f}, "
                f"got {observed_ratio:.2f} (count: {dataset_counts[dataset_name]})"
            )

    def test_metrics_aggregation(self, dataset_factory, small_dataset_file, medium_dataset_file):
        """Tests that metrics from all child datasets are collected and aggregated."""
        # Create two datasets with different characteristics
        ds1 = dataset_factory(small_dataset_file, dataset_name="ds1")
        ds2 = dataset_factory(medium_dataset_file, dataset_name="ds2")

        interleaved = InterleavedDataset([ds1, ds2], [0.2, 0.8], seed=SEED)
        aggregator = MetricsAggregator()

        # Process some samples
        for sample in islice(iter(interleaved), 50):
            aggregator.update(sample["metrics"])

        metrics = aggregator.get_metrics()

        # Should have metrics from both datasets
        assert_metrics_structure(metrics, ["dataset_1", "dataset_2"])

        # Both datasets should have contributed samples
        assert metrics["dataset_1"]["samples_seen"] > 0
        assert metrics["dataset_2"]["samples_seen"] > 0

        # Total samples should equal what we processed
        total_samples = (
            metrics["dataset_1"]["samples_seen"] + metrics["dataset_2"]["samples_seen"]
        )
        assert total_samples == 50

    def test_checkpointing(self, dataset_factory, tmp_data_dir):
        """Tests that interleaved dataset checkpointing preserves sampling state."""

        def create_interleaved():
            ds1 = dataset_factory(small_dataset_file, dataset_name="ds1")
            ds2 = dataset_factory(medium_dataset_file, dataset_name="ds2")
            return InterleavedDataset([ds1, ds2], [0.7, 0.3], seed=SEED)

        # Original run
        interleaved1 = create_interleaved()
        iterator1 = iter(interleaved1)

        # Consume samples and checkpoint
        consumed = list(islice(iterator1, 30))
        state = interleaved1.state_dict()

        # Create new instance and restore
        interleaved2 = create_interleaved()
        interleaved2.load_state_dict(state)

        # Verify sampling pattern continues correctly
        assert_checkpoint_continuation(iterator1, interleaved2, num_samples=10)


class TestEndToEndCheckpointing:
    """
    End-to-end tests for the full data loading and checkpointing pipeline,
    including `StatefulDataLoader`, the datasets, and `MetricsAggregator`.
    """

    @pytest.mark.parametrize("dataset_type", ["single", "interleaved"])
    @pytest.mark.parametrize("shuffle", [False, True])
    @pytest.mark.parametrize("num_workers", [0, 3])
    @pytest.mark.parametrize(
        "checkpoint_timing", ["very_early", "mid_epoch", "epoch_boundary"]
    )
    def test_full_pipeline_checkpointing(
        self,
        dataset_type,
        shuffle,
        num_workers,
        checkpoint_timing,
        dataset_factory,
        tmp_data_dir,
    ):
        """
        Test the complete data loading pipeline with checkpointing.
        This is the most critical test as it validates real training scenarios.
        """
        # Run all combinations - don't skip any

        def create_pipeline():
            """Create the complete data loading pipeline."""
            if dataset_type == "single":
                file1 = tmp_data_dir / "single_data.json"
                create_test_json_file(file1, LARGE_DATASET_SIZE)
                dataset = dataset_factory(str(file1), shuffle=shuffle)
            elif dataset_type == "interleaved":
                file1 = tmp_data_dir / "interleaved_1.json"
                file2 = tmp_data_dir / "interleaved_2.json"
                create_test_json_file(file1, MEDIUM_DATASET_SIZE)
                create_test_json_file(file2, MEDIUM_DATASET_SIZE, offset=1000)

                ds1 = dataset_factory(str(file1), dataset_name="ds1", shuffle=shuffle)
                ds2 = dataset_factory(str(file2), dataset_name="ds2", shuffle=shuffle)
                dataset = InterleavedDataset([ds1, ds2], [0.5, 0.5], seed=SEED)

            # in_order=True: batches returned in FIFO order (deterministic when possible)
            # in_order=False: batches can be returned out of order (faster but non-deterministic)
            # For non-shuffled data with multiple workers, we want deterministic order
            # For shuffled data, order doesn't matter so we can use faster out-of-order processing
            use_in_order = shuffle or num_workers <= 1

            loader = StatefulDataLoader(
                dataset,
                batch_size=BATCH_SIZE,
                num_workers=num_workers,
                collate_fn=collate_with_metrics,
                in_order=use_in_order,
            )

            return loader, MetricsAggregator()

        # Determine checkpoint step based on timing
        samples_per_epoch = (
            LARGE_DATASET_SIZE if dataset_type == "single" else MEDIUM_DATASET_SIZE * 2
        )
        batches_per_epoch = samples_per_epoch // BATCH_SIZE

        if checkpoint_timing == "very_early":
            steps_before_checkpoint = 2  # After just 1 batch
        elif checkpoint_timing == "mid_epoch":
            steps_before_checkpoint = max(
                1, batches_per_epoch // 3
            )  # Not a clean fraction
        else:  # epoch_boundary
            steps_before_checkpoint = batches_per_epoch

        # === PHASE 1: Original run to checkpoint ===
        loader1, aggregator1 = create_pipeline()
        iter_loader1 = iter(loader1)
        # Run training loop until checkpoint
        pre_checkpoint_batches, pre_checkpoint_metrics = run_training_loop(
            iter_loader1, aggregator1, steps_before_checkpoint
        )

        # Save complete state
        loader_state = loader1.state_dict()
        aggregator_state = aggregator1.state_dict()

        # Continue for ground truth - create new iterator after checkpoint
        continuation_batches, continuation_metrics = run_training_loop(
            iter_loader1, aggregator1, 5
        )

        # === PHASE 2: Restored run from checkpoint ===
        loader2, aggregator2 = create_pipeline()

        # Restore state
        loader2.load_state_dict(loader_state)
        aggregator2.load_state_dict(aggregator_state)

        # Verify aggregator state was restored correctly
        restored_pre_metrics = aggregator2.get_metrics()
        assert (
            restored_pre_metrics == pre_checkpoint_metrics
        ), "Aggregator state not restored correctly"

        # Run same number of steps
        resumed_batches, resumed_metrics = run_training_loop(
            iter(loader2), aggregator2, 5
        )

        # === ASSERTIONS ===

        assert len(continuation_batches) == len(resumed_batches), "Batch count mismatch"

        for i, (orig_batch, restored_batch) in enumerate(
            zip(continuation_batches, resumed_batches)
        ):
            assert (
                orig_batch["id"].shape == restored_batch["id"].shape
            ), f"Batch {i} shape mismatch: {orig_batch['id'].shape} vs {restored_batch['id'].shape}"

        # For non-shuffled data, batches should be identical
        if not shuffle:
            continuation_ids = torch.cat(
                [b["id"] for b in continuation_batches]
            ).tolist()
            resumed_ids = torch.cat([b["id"] for b in resumed_batches]).tolist()

            assert continuation_ids == resumed_ids, (
                f"Non-shuffled data should have identical continuation.\n"
                f"pre_checkpoint_batches: {[b['id'].tolist() for b in pre_checkpoint_batches]}\n"
                f"Original continuation: {continuation_ids}\n"
                f"Resumed continuation:  {resumed_ids}"
            )

            # Metrics should be identical for non-shuffled data.
            assert continuation_metrics == resumed_metrics, (
                f"Metrics mismatch after checkpoint restore for non-shuffled data.\n"
                f"Original: {continuation_metrics}\n"
                f"Resumed:  {resumed_metrics}"
            )
        else:
            # For shuffled data, we can't assert that metrics are identical since
            # the sample order will differ. Instead, we perform a basic sanity check
            # on the structure and presence of expected keys.
            expected_datasets = (
                ["test_dataset"] if dataset_type == "single" else ["ds1", "ds2"]
            )
            assert_metrics_structure(resumed_metrics, expected_datasets)


class TestDistributedMetricsAggregator(FSDPTest):
    """Distributed tests for MetricsAggregator using FSDPTest infrastructure."""

    @property
    def world_size(self) -> int:
        return 2

    def setUp(self):
        super().setUp()
        # Use a rank-specific temp directory to avoid race conditions
        # when tests are run in parallel on the same file system.
        self.tmp_dir = tempfile.TemporaryDirectory(prefix=f"rank_{self.rank}_")
        self.tmp_path = Path(self.tmp_dir.name)

    def tearDown(self):
        self.tmp_dir.cleanup()
        super().tearDown()

    @gpu_test(gpu_count=2)
    def test_distributed_aggregation(self):
        """Test metrics aggregation across multiple ranks."""
        rank = dist.get_rank()
        world_size = dist.get_world_size()

        aggregator = MetricsAggregator()

        # Each rank contributes different values
        metrics = [
            Metric("test", "rank_sum", rank * 10, AggregationType.SUM),
            Metric("test", "rank_max", rank * 5, AggregationType.MAX),
            Metric("test", "sample_count", 1, AggregationType.SUM),
        ]
        aggregator.update(metrics)

        result = aggregator.get_metrics()

        # Verify distributed aggregation
        expected_sum = sum(
            r * 10 for r in range(world_size)
        )  # 0*10 + 1*10 = 10 for world_size=2
        expected_max = (world_size - 1) * 5  # max(0*5, 1*5) = 5 for world_size=2

        assert result["test"]["rank_sum"] == expected_sum
        assert result["test"]["rank_max"] == expected_max
        assert result["test"]["sample_count"] == world_size


class TestDistributedDataLoading(FSDPTest):
    """Distributed tests for the data loading pipeline using FSDPTest."""

    @property
    def world_size(self) -> int:
        return 2

    def setUp(self):
        super().setUp()
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self.tmp_dir.name)

    def tearDown(self):
        self.tmp_dir.cleanup()
        super().tearDown()

    def dataset_factory(
        self,
        data_file: str,
        dataset_name: str = "test_dataset",
        shuffle: bool = False,
        message_transform: Optional[Callable] = None,
        **kwargs,
    ) -> HFIterableDataset:
        """Factory for creating HFIterableDataset instances with common defaults."""
        return HFIterableDataset(
            path="json",
            data_files=data_file,
            split="train",
            dataset_name=dataset_name,
            seed=SEED,
            shuffle_buffer_size=DEFAULT_SHUFFLE_BUFFER_SIZE if shuffle else 0,
            message_transform=message_transform or (lambda x: x),
            model_transform=lambda x: x,
            output_transform=lambda x: x,
            metric_transform=StandardMetricTransform(dataset_name),
            **kwargs,
        )

    @gpu_test(gpu_count=2)
    def test_distributed_basic_loading(self):
        """Tests that data is correctly sharded across ranks in a distributed setting."""
        rank = dist.get_rank()
        world_size = dist.get_world_size()

        # Create shared dataset file
        data_file = self.tmp_path / "distributed_data.json"
        create_test_json_file(data_file, LARGE_DATASET_SIZE)

        dataset = self.dataset_factory(str(data_file))
        loader = DataLoader(
            dataset,
            batch_size=BATCH_SIZE,
            num_workers=0,
            collate_fn=collate_with_metrics,
        )

        # Collect all samples seen by this rank
        all_ids_this_rank = []
        for batch in islice(loader, 10):  # Just first 10 batches
            all_ids_this_rank.extend(batch["id"].tolist())

        # Gather all IDs from all ranks
        all_rank_ids = [None] * world_size
        dist.all_gather_object(all_rank_ids, all_ids_this_rank)

        # Verify no overlap between ranks (proper sharding)
        rank0_ids = set(all_rank_ids[0])
        rank1_ids = set(all_rank_ids[1])

        assert rank0_ids.isdisjoint(
            rank1_ids
        ), f"Ranks should not see overlapping data. Overlap: {rank0_ids & rank1_ids}"

        # Verify both ranks got data
        assert len(rank0_ids) > 0, "Rank 0 should have processed some data"
        assert len(rank1_ids) > 0, "Rank 1 should have processed some data"

    @gpu_test(gpu_count=2)
    @pytest.mark.parametrize("num_workers", [0, 2])
    def test_distributed_with_workers(self, num_workers):
        """Test distributed loading with multiple workers per rank."""
        rank = dist.get_rank()
        world_size = dist.get_world_size()

        # Create dataset file
        data_file = self.tmp_path / f"distributed_worker_data.json"
        create_test_json_file(data_file, LARGE_DATASET_SIZE)

        dataset = self.dataset_factory(str(data_file), shuffle=True)
        loader = StatefulDataLoader(
            dataset,
            batch_size=BATCH_SIZE,
            num_workers=num_workers,
            collate_fn=collate_with_metrics,
        )
        aggregator = MetricsAggregator()

        # Process some batches
        steps = 15
        for batch in islice(loader, steps):
            aggregator.update(batch.pop("metrics"))

        # Get metrics - this includes distributed reduction
        metrics = aggregator.get_metrics()

        # Verify total samples processed across all ranks
        total_samples = metrics["test_dataset"]["samples_seen"]
        expected_min_samples = (
            steps * BATCH_SIZE * world_size * 0.8
        )  # Allow 20% variance

        assert (
            total_samples >= expected_min_samples
        ), f"Expected at least {expected_min_samples} total samples across ranks, got {total_samples}"

    @gpu_test(gpu_count=2)
    def test_distributed_interleaved_dataset(self):
        """Tests the interleaved dataset in a distributed setting, checking metrics aggregation."""
        rank = dist.get_rank()

        # Create dataset files
        file1 = self.tmp_path / "dist_ds1.json"
        file2 = self.tmp_path / "dist_ds2.json"
        create_test_json_file(file1, MEDIUM_DATASET_SIZE)
        create_test_json_file(file2, MEDIUM_DATASET_SIZE, offset=1000)

        ds1 = self.dataset_factory(str(file1), dataset_name="dist_ds1", shuffle=True)
        ds2 = self.dataset_factory(str(file2), dataset_name="dist_ds2", shuffle=True)

        interleaved = InterleavedDataset([ds1, ds2], [0.7, 0.3], seed=SEED)
        loader = StatefulDataLoader(
            interleaved,
            batch_size=BATCH_SIZE,
            num_workers=0,
            collate_fn=collate_with_metrics,
        )
        aggregator = MetricsAggregator()

        # Process samples
        for batch in islice(loader, 20):
            aggregator.update(batch.pop("metrics"))

        # Get global metrics
        metrics = aggregator.get_metrics()

        # Verify both datasets contributed across all ranks
        assert "dist_ds1" in metrics
        assert "dist_ds2" in metrics

        total_ds1 = metrics["dist_ds1"]["samples_seen"]
        total_ds2 = metrics["dist_ds2"]["samples_seen"]
        total_samples = total_ds1 + total_ds2

        # Verify sampling ratios are maintained globally
        observed_ratio_ds1 = total_ds1 / total_samples
        assert (
            abs(observed_ratio_ds1 - 0.7) < 0.1
        ), f"Global sampling ratio off: expected 0.7, got {observed_ratio_ds1:.2f}"

    @gpu_test(gpu_count=2)
    def test_distributed_checkpointing_with_workers(self):
        """Test checkpointing in distributed setting with multiple workers."""
        rank = dist.get_rank()

        # Create dataset
        data_file = self.tmp_path / "dist_checkpoint_data.json"
        create_test_json_file(data_file, LARGE_DATASET_SIZE)

        def create_loader():
            dataset = self.dataset_factory(str(data_file), shuffle=True)
            return (
                StatefulDataLoader(
                    dataset,
                    batch_size=BATCH_SIZE,
                    num_workers=2,
                    collate_fn=collate_with_metrics,
                ),
                MetricsAggregator(),
            )

        # Original run
        loader1, aggregator1 = create_loader()

        # Process some batches
        for batch in islice(loader1, 10):
            aggregator1.update(batch.pop("metrics"))

        # Checkpoint
        loader_state = loader1.state_dict()
        aggregator_state = aggregator1.state_dict()
        metrics_at_checkpoint = aggregator1.get_metrics()

        # Continue original
        continuation_ids = []
        for batch in islice(loader1, 5):
            continuation_ids.extend(batch["id"].tolist())
            aggregator1.update(batch.pop("metrics"))
        final_metrics = aggregator1.get_metrics()

        # Create new loader and restore
        loader2, aggregator2 = create_loader()
        loader2.load_state_dict(loader_state)
        aggregator2.load_state_dict(aggregator_state)

        # Verify metrics were restored
        restored_metrics = aggregator2.get_metrics()
        assert restored_metrics == metrics_at_checkpoint

        # Continue with restored loader
        resumed_ids = []
        for batch in islice(loader2, 5):
            resumed_ids.extend(batch["id"].tolist())
            aggregator2.update(batch.pop("metrics"))
        resumed_final_metrics = aggregator2.get_metrics()

        # Metrics should match (they include distributed aggregation)
        assert final_metrics == resumed_final_metrics

        # For shuffled data, exact sample order may differ but metrics should be consistent
        assert len(continuation_ids) == len(resumed_ids)

# --------------------------------------------------------------------------------
# Performance and Edge Case Tests
# --------------------------------------------------------------------------------


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_empty_dataset_handling(self, dataset_factory, tmp_data_dir):
        """Tests that an empty dataset is handled without errors."""
        # Create empty dataset file
        empty_file = tmp_data_dir / "empty.json"
        empty_file.write_text("")

        dataset = dataset_factory(str(empty_file))

        # Should handle empty dataset gracefully
        samples = list(islice(iter(dataset), 5))

        # May get no samples (which is fine) or raise informative error
        if len(samples) == 0:
            # This is acceptable behavior
            pass
        else:
            # If we do get samples, they should be well-formed
            for sample in samples:
                assert_sample_structure(sample, "Empty dataset sample")

    def test_dataset_with_variable_fields(self, dataset_factory, tmp_data_dir):
        """Test handling of samples with different fields (schema evolution)."""
        # Create dataset with evolving schema
        evolving_file = tmp_data_dir / "evolving.json"
        with open(evolving_file, "w") as f:
            # First 10 samples have basic fields
            for i in range(10):
                f.write(f'{{"id": {i}, "tokens": [{i}, {i+1}], "text": "basic_{i}"}}\n')
            # Next 10 have additional field
            for i in range(10, 20):
                f.write(
                    f'{{"id": {i}, "tokens": [{i}, {i+1}], "text": "extended_{i}", "extra": "data_{i}"}}\n'
                )
            # Last 10 have different additional field
            for i in range(20, 30):
                f.write(
                    f'{{"id": {i}, "tokens": [{i}, {i+1}], "text": "modified_{i}", "other": {i*2}}}\n'
                )

        dataset = dataset_factory(str(evolving_file))
        loader = DataLoader(dataset, batch_size=5, collate_fn=collate_with_metrics)

        # Process all batches - should handle variable schemas
        batches_processed = 0
        for batch in loader:
            batches_processed += 1
            # Basic fields should always be present
            assert "id" in batch
            assert "tokens" in batch
            assert "metrics" in batch

            # Batch size should be consistent (except possibly last batch)
            assert batch["id"].shape[0] <= 5

        assert (
            batches_processed == 6
        ), f"Expected 6 batches from 30 samples, got {batches_processed}"
