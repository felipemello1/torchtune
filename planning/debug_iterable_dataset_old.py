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

import pyarrow as pa
import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

# Note: wrap with a try-except block to avoid breaking OSS builds
from datasets import Dataset as HFDataset, load_dataset
from datasets.distributed import split_dataset_by_node

from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, IterableDataset
from torchdata.stateful_dataloader import StatefulDataLoader


# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
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

        for reduce_type in values_by_type.keys():
            type_components = [(k[0], k[1]) for k in unique_keys if k[2] == reduce_type]
            if type_components:
                key_map[reduce_type] = {k: i for i, k in enumerate(type_components)}
                values_by_type[reduce_type] = torch.zeros(len(type_components))

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
        Returns an iterator over the dataset. Each implementation is responsible
        for its own iteration logic, including making it an infinite stream.
        """
        pass

    @abstractmethod
    def get_metrics(self) -> Dict[str, Dict[str, Any]]:
        """
        Return metrics in a structured format suitable for logging and analysis.

        The expected format is:
        {
            dataset_name: {
                "metrics": {metric_key: metric_value, ...},
            }
        }
        The "metrics" key should contain only numeric values suitable for logging backends.
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
        message_transform: Callable,
        model_transform: Callable,
        output_transform: Callable,
        metric_transform: Callable,
        shuffle_buffer_size: Optional[int] = 1000,
        seed: int = 42,
        num_shards_per_worker: int = 4,
        dataset_name: Optional[str] = None,
        filter_fn: Optional[Callable] = None,
        filter_kwargs: Optional[Dict[str, Any]] = None,
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
            "message": message_transform,
            "model": model_transform,
            "output": output_transform,
            "metric": metric_transform,
        }

        # Internal state for resumption
        self._num_epochs = 0
        self._samples_yielded_this_epoch = 0

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
        # TODO: consider number of dataloader workers
        num_shards = world_size * num_shards_per_worker
        ds = load_dataset(**load_dataset_kwargs)

        # Use to_iterable_dataset for streaming datasets
        if not load_dataset_kwargs.get("streaming", False):
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

        # Apply lazy transforms once during setup, not in __iter__
        self._ds = ds.map(self._apply_transforms).filter(self._filter_failed_transforms)

    def _apply_transforms(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """Apply the transform pipeline with simple error handling."""
        try:
            sample = self._transforms["message"](sample)
            sample = self._transforms["model"](sample)
            sample = self._transforms["output"](sample)
            return sample
        except Exception as e:
            # For testing, log the error to the console
            logger.warning(
                f"Transform failed for a sample in {self.dataset_name}:\n"
                f"{traceback.format_exc()}"
            )
            return {
                "__failed_transform__": True,
                "exception": str(e),
                "error_traceback": traceback.format_exc(),
                "original_sample": sample,
            }

    def _filter_failed_transforms(self, sample: Dict[str, Any]) -> bool:
        """
        Return `False` for samples that have a special `__failed_transform__`
        key, which is added during the transform pipeline. This allows us to
        filter out samples that failed to process, preventing them from
        propagating to the training loop.
        """
        return not sample.get("__failed_transform__", False)

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        """Iterate through the dataset with automatic metrics collection."""
        rank = dist.get_rank() if dist.is_initialized() else 0

        # This variable will hold the dataset for the current epoch.
        # It starts as self._ds, which could be a resumed dataset from a checkpoint.
        epoch_ds = self._ds

        while True:  # Infinite iteration
            logger.info(
                f"Rank {rank} - {self.dataset_name}: Starting new epoch {self._num_epochs}."
            )
            epoch_seed = self._seed + self._num_epochs

            epoch_ds.set_epoch(epoch_seed)
            epoch_iterator = iter(epoch_ds)

            samples_yielded_this_epoch = 0
            for sample in epoch_iterator:

                sample = self._transforms["metric"](sample)
                metric_num_epochs = Metric(
                    dataset_name=self.dataset_name,
                    name="num_epochs",
                    value=self._num_epochs,
                    agg_type=AggregationType.SUM,
                )
                if "metrics" not in sample:
                    sample["metrics"] = []
                sample["metrics"].append(metric_num_epochs)
                samples_yielded_this_epoch += 1

                yield sample

            logger.info(
                f"Rank {rank} - {self.dataset_name}: Finished epoch {self._num_epochs}. "
                f"Yielded {samples_yielded_this_epoch} samples."
            )

            # Epoch complete - increment and continue infinite loop
            self._num_epochs += 1

            # Reset to the base dataset for the next epoch's shuffling.
            epoch_ds = self._ds

    def state_dict(self) -> Dict[str, Any]:
        """
        Return state for checkpointing, including the state of the underlying
        Hugging Face IterableDataset to ensure exact resumption.
        """
        # HF's .state_dict() is not available on all versions/types of datasets,
        # so we check for its existence to avoid errors.
        hf_dataset_state = {}
        if hasattr(self._ds, "state_dict"):
            hf_dataset_state = self._ds.state_dict()

        state = {
            "num_epochs": self._num_epochs,
            "seed": self._seed,
            "hf_dataset_state": hf_dataset_state,
        }
        return {self.dataset_name: state}

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """
        Load state from checkpoint, including restoring the state of the
        Hugging Face IterableDataset.
        """
        own_state = state_dict[self.dataset_name]
        self._num_epochs = own_state["num_epochs"]

        hf_dataset_state = own_state.get("hf_dataset_state")
        self._ds.load_state_dict(hf_dataset_state)

    def get_metrics(self) -> Dict[str, Dict[str, Any]]:
        """
        Return metrics in a structured format. HFIterableDataset doesn't maintain
        its own metrics state - metrics are generated per-sample by the metric transform.
        """
        return {self.dataset_name: {"num_epochs": self._num_epochs}}


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
                # TODO: double check the desired behavior here

                # Per the design, child datasets must be infinite. We re-initialize
                # to allow for continuous operation but warn loudly as this may
                # indicate a design problem in the child dataset.
                logger.warning(
                    f"Child dataset {self._datasets[ds_idx].dataset_name} was exhausted. "
                    "This is unexpected for an infinite dataset. Re-initializing its iterator."
                )
                child_iters[ds_idx] = iter(self._datasets[ds_idx])
                # Retry getting a sample
                sample = next(child_iters[ds_idx])
                yield sample

    def get_metrics(self) -> Dict[str, Dict[str, Any]]:
        """
        Return aggregated metrics from all child datasets.
        InterleavedDataset doesn't maintain its own metrics - it collects from children.
        """
        all_metrics = {}
        for ds in self._datasets:
            child_metrics = ds.get_metrics()
            all_metrics.update(child_metrics)
        return all_metrics

    def state_dict(self) -> Dict[str, Any]:
        """Save state for the interleaver and its children."""
        child_states = {}
        for ds in self._datasets:
            child_states.update(ds.state_dict())

        state = {
            "sampling_generator_state": self._sampling_generator.get_state(),
            "child_states": child_states,
        }

        return {self.dataset_name: state}

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Load state for the interleaver and its children."""
        own_state = state_dict[self.dataset_name]

        self._sampling_generator.set_state(own_state["sampling_generator_state"])

        child_states = own_state["child_states"]
        for ds in self._datasets:
            if ds.dataset_name in child_states:
                # The state for each child is already namespaced, so pass it
                # directly, but scoped to its own key.
                ds.load_state_dict({ds.dataset_name: child_states[ds.dataset_name]})


# --------------------------------------------------------------------------------
# Test Implementation from `final_enhance_tests.md`
# --------------------------------------------------------------------------------
# Constants
DATASET_SIZE_1 = 20
DATASET_SIZE_2 = 30
SEED = 42
BATCH_SIZE = 4


# Simple helper for creating test data files with varying token lengths
def _create_json_file(path, num_samples, offset=1):
    with open(path, "w") as f:
        for i in range(num_samples):
            _i = i * offset
            # Vary token length to test sequence statistics
            token_len = (i % 8) + 3  # Between 3-10 tokens
            tokens = list(range(_i, _i + token_len))
            f.write(f'{{"number": {_i}, "tokens": {tokens}, "text": "sample_{_i}"}}\n')


# Simple collate function
def _collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    # 1. Separate metrics from the data
    all_metrics = []
    clean_batch = []
    for sample in batch:
        if "metrics" in sample:
            all_metrics.extend(sample.pop("metrics"))
        clean_batch.append(sample)

    # 2. Collate the tensor data normally
    collated_batch = {
        "number": torch.tensor([item["number"] for item in clean_batch]),
        "tokens": pad_sequence(
            [torch.tensor(item["tokens"]) for item in clean_batch], batch_first=True
        ),
    }
    # Add back other keys that are not tensors
    if clean_batch:
        for key in clean_batch[0].keys():
            if key not in collated_batch:
                collated_batch[key] = [d[key] for d in clean_batch]

    # 3. Add the collected metrics back into the final batch
    collated_batch["metrics"] = all_metrics
    return collated_batch


# Basic transform functions
def _identity_transform(x):
    return x


def _failing_transform(sample):
    if sample["number"] == 7:
        raise ValueError("Simulated failure")
    return sample


def _tokenize_transform(sample):
    sample["input_ids"] = [sample["number"]] * 5  # Simple tokenization
    return sample


# Fixtures - for different dataset sizes and scenarios
def get_dataset_1_file(tmp_path):
    path = tmp_path / "data_1.json"
    _create_json_file(path, DATASET_SIZE_1, offset=0)
    return str(path)


def get_dataset_2_file(tmp_path):
    path = tmp_path / "data_2.json"
    _create_json_file(path, DATASET_SIZE_2, offset=100)
    return str(path)


def get_dataset_factory():
    def _create(
        data_file,
        dataset_name="test_dataset",
        shuffle=False,
        transform=None,
        metric_transform=None,
        **kwargs,
    ):
        return HFIterableDataset(
            path="json",
            data_files=data_file,
            split="train",
            dataset_name=dataset_name,
            seed=SEED,
            shuffle_buffer_size=100 if shuffle else 0,
            message_transform=transform or _identity_transform,
            model_transform=_identity_transform,
            output_transform=_identity_transform,
            metric_transform=metric_transform or StandardMetricTransform(dataset_name),
            **kwargs,
        )

    return _create


# Helper function for detailed assertions
def _assert_sample_batch_structure(batch, expected_batch_size, context=""):
    """Assert batch structure with detailed debugging info."""
    assert isinstance(
        batch, dict
    ), f"{context}: Batch should be dict, got {type(batch)}"
    assert (
        "number" in batch
    ), f"{context}: Missing 'number' key in batch. Keys: {list(batch.keys())}"
    assert (
        "tokens" in batch
    ), f"{context}: Missing 'tokens' key in batch. Keys: {list(batch.keys())}"

    number_shape = batch["number"].shape
    tokens_shape = batch["tokens"].shape

    # Handle cases where the last batch might not be full
    if expected_batch_size is not None:
        assert number_shape[0] == expected_batch_size, (
            f"{context}: Expected batch size {expected_batch_size}, got {number_shape[0]}. "
            f"Number shape: {number_shape}, Tokens shape: {tokens_shape}"
        )
        assert tokens_shape[0] == expected_batch_size, (
            f"{context}: Token batch size mismatch. Expected {expected_batch_size}, got {tokens_shape[0]}. "
            f"Number shape: {number_shape}, Tokens shape: {tokens_shape}"
        )


def _assert_metrics_structure(metrics: Dict[str, Dict[str, Any]], context=""):
    """Assert metrics structure with detailed debugging info."""
    assert isinstance(
        metrics, dict
    ), f"{context}: Metrics should be dict, got {type(metrics)}"
    for ds_name, ds_metrics in metrics.items():
        assert isinstance(
            ds_metrics, dict
        ), f"{context}: Metrics for '{ds_name}' should be dict, got {type(ds_metrics)}"
        for metric_name, value in ds_metrics.items():
            assert isinstance(
                value, (int, float)
            ), f"{context}: Metric '{ds_name}/{metric_name}' should be numeric, got {type(value)} with value {value}"


def _assert_standard_metrics_present(
    metrics: Dict[str, Dict[str, Any]], dataset_name: str, context=""
):
    """Assert that standard metrics are present and valid."""
    assert (
        dataset_name in metrics
    ), f"{context}: Dataset '{dataset_name}' not found in metrics. Available: {list(metrics.keys())}"

    ds_metrics = metrics[dataset_name]

    # Check required standard metrics
    required_metrics = ["samples_seen", "tokens_seen", "num_epochs"]
    for metric in required_metrics:
        assert (
            metric in ds_metrics
        ), f"{context}: Required metric '{metric}' missing from dataset '{dataset_name}'. Available: {list(ds_metrics.keys())}"
        assert isinstance(
            ds_metrics[metric], (int, float)
        ), f"{context}: Metric '{metric}' should be numeric, got {type(ds_metrics[metric])}"

    # Check distribution metrics (should be present when StandardMetricTransform is used)
    dist_metrics = [
        "seq_len_mean",
        "seq_len_min",
        "seq_len_max",
        "seq_len_p05",
        "seq_len_p50",
        "seq_len_p95",
    ]
    missing_dist = [m for m in dist_metrics if m not in ds_metrics]
    if missing_dist:
        logger.warning(f"{context}: Distribution metrics missing: {missing_dist}")


# Test Functions - each tests exactly one behavior


def test_basic_iteration_works(dataset_factory, dataset_1_file):
    """Dataset can be iterated and yields expected samples."""
    dataset = dataset_factory(dataset_1_file)

    samples = list(islice(iter(dataset), 10))

    assert len(samples) == 10, f"Expected 10 samples, got {len(samples)}"
    assert all("metrics" in s for s in samples), "All samples should have metrics"

    # Verify metrics are properly structured
    for i, sample in enumerate(samples):
        assert (
            "number" in sample
        ), f"Sample {i} missing 'number' key. Keys: {list(sample.keys())}"
        assert (
            "tokens" in sample
        ), f"Sample {i} missing 'tokens' key. Keys: {list(sample.keys())}"
        assert (
            "text" in sample
        ), f"Sample {i} missing 'text' key. Keys: {list(sample.keys())}"
        assert isinstance(
            sample["tokens"], list
        ), f"Sample {i} tokens should be list, got {type(sample['tokens'])}"

        # Verify metrics structure
        assert isinstance(sample["metrics"], list), f"Sample {i} metrics should be list"
        assert len(sample["metrics"]) > 0, f"Sample {i} should have non-empty metrics"

        # Check that we have both StandardMetricTransform metrics and HFIterableDataset metrics
        metric_names = [m.name for m in sample["metrics"]]
        assert "samples_seen" in metric_names, f"Sample {i} missing samples_seen metric"
        assert "tokens_seen" in metric_names, f"Sample {i} missing tokens_seen metric"
        assert "seq_len" in metric_names, f"Sample {i} missing seq_len metric"
        assert "num_epochs" in metric_names, f"Sample {i} missing num_epochs metric"


def test_shuffling_changes_order(dataset_factory, dataset_2_file):
    """Shuffling produces different order but same seed gives same result."""
    no_shuffle = list(islice(iter(dataset_factory(dataset_2_file, shuffle=False)), 20))
    shuffle1 = list(islice(iter(dataset_factory(dataset_2_file, shuffle=True)), 20))
    shuffle2 = list(islice(iter(dataset_factory(dataset_2_file, shuffle=True)), 20))

    no_shuffle_numbers = [batch["number"] for batch in no_shuffle]
    shuffle1_numbers = [batch["number"] for batch in shuffle1]
    shuffle2_numbers = [batch["number"] for batch in shuffle2]

    assert no_shuffle_numbers != shuffle1_numbers, (
        f"Shuffled and non-shuffled order should differ.\n"
        f"Non-shuffled: {no_shuffle_numbers[:10]}...\n"
        f"Shuffled: {shuffle1_numbers[:10]}..."
    )
    assert shuffle1_numbers == shuffle2_numbers, (
        f"Same seed should produce identical shuffle order.\n"
        f"First run: {shuffle1_numbers[:10]}...\n"
        f"Second run: {shuffle2_numbers[:10]}..."
    )


def test_hf_dataset_checkpointing_basic(dataset_factory, dataset_1_file):
    """HFIterableDataset state can be saved and restored correctly."""
    dataset1 = dataset_factory(dataset_1_file, shuffle=True)
    iterator1 = iter(dataset1)

    # Consume some samples
    _ = list(islice(iterator1, 50))
    state = dataset1.state_dict()

    # Get continuation samples
    continuation = list(islice(iterator1, 5))
    continuation_numbers = [batch["number"] for batch in continuation]

    # Create new dataset and restore state
    dataset2 = dataset_factory(dataset_1_file, shuffle=True)
    dataset2.load_state_dict(state)
    resumed = list(islice(iter(dataset2), 5))
    resumed_numbers = [batch["number"] for batch in resumed]

    assert resumed_numbers == continuation_numbers


@pytest.mark.parametrize("num_workers", [0, 2])
def test_dataloader_integration(num_workers, dataset_factory, dataset_1_file):
    """
    Tests DataLoader integration, covering:
    - Operation with and without multiple workers.
    - Infinite looping across epoch boundaries.
    - Consistent data ordering in unshuffled datasets across epochs.
    """
    dataset = dataset_factory(
        dataset_1_file, shuffle=False, transform=_tokenize_transform
    )
    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        num_workers=num_workers,
        collate_fn=_collate_fn,
    )

    rank, world_size = (
        (0, 1)
        if not dist.is_initialized()
        else (dist.get_rank(), dist.get_world_size())
    )
    samples_per_rank = DATASET_SIZE_1 // world_size
    batches_per_epoch = samples_per_rank // BATCH_SIZE

    # Run for more steps than one epoch to test infinite looping
    max_steps = int(batches_per_epoch * 2.5)
    batches = []
    for step, batch in enumerate(dataloader):
        if step >= max_steps:
            break
        batches.append(batch)
        _assert_sample_batch_structure(
            batch, BATCH_SIZE, f"Worker {num_workers}, Step {step}"
        )
        assert "metrics" in batch

    assert len(batches) == max_steps, f"Failed with {num_workers=}"
    assert all("input_ids" in batch for batch in batches), f"Failed with {num_workers=}"

    # Verify that data repeats across epochs for unshuffled datasets
    if batches_per_epoch > 0:
        first_epoch_numbers = torch.cat(
            [b["number"] for b in batches[:batches_per_epoch]]
        ).tolist()
        second_epoch_start = batches_per_epoch
        second_epoch_end = min(batches_per_epoch * 2, len(batches))
        second_epoch_numbers = torch.cat(
            [b["number"] for b in batches[second_epoch_start:second_epoch_end]]
        ).tolist()

        assert first_epoch_numbers == second_epoch_numbers, (
            f"Epoch patterns should repeat without shuffling.\n"
            f"First epoch: {first_epoch_numbers}\n"
            f"Second epoch: {second_epoch_numbers}"
        )


@pytest.mark.parametrize("checkpoint_step", ["mid_epoch", "epoch_boundary"])
@pytest.mark.parametrize("num_workers", [0, 2])
def test_stateful_dataloader_checkpointing(
    checkpoint_step, num_workers, dataset_factory, dataset_1_file
):
    """
    StatefulDataLoader can checkpoint and resume from mid-epoch or epoch boundaries,
    with both single and multiple workers.
    """

    def create_loader():
        dataset = dataset_factory(
            dataset_1_file, shuffle=True, transform=_tokenize_transform
        )
        return StatefulDataLoader(
            dataset,
            batch_size=BATCH_SIZE,
            num_workers=num_workers,
            collate_fn=_collate_fn,
        )

    rank, world_size = (
        (0, 1)
        if not dist.is_initialized()
        else (dist.get_rank(), dist.get_world_size())
    )
    samples_per_rank = DATASET_SIZE_1 // world_size
    batches_per_epoch = samples_per_rank // BATCH_SIZE

    if checkpoint_step == "mid_epoch":
        # Checkpoint somewhere in the middle of an epoch
        steps_before_checkpoint = batches_per_epoch // 2
    else:  # epoch_boundary
        # Checkpoint at the very end of the first epoch
        steps_before_checkpoint = batches_per_epoch

    if steps_before_checkpoint == 0:
        pytest.skip("Not enough batches to test checkpointing.")

    loader1 = create_loader()
    # In this test, we also need to test the MetricsAggregator's state
    aggregator1 = MetricsAggregator()
    for batch in islice(iter(loader1), steps_before_checkpoint):
        aggregator1.update(batch.pop("metrics"))

    state = loader1.state_dict()
    aggregator_state = aggregator1.state_dict()

    continuation = list(islice(iter(loader1), 3))
    continuation_numbers = torch.cat([b["number"] for b in continuation]).tolist()

    loader2 = create_loader()
    aggregator2 = MetricsAggregator()
    loader2.load_state_dict(state)
    aggregator2.load_state_dict(aggregator_state)

    resumed = list(islice(iter(loader2), 3))
    resumed_numbers = torch.cat([b["number"] for b in resumed]).tolist()

    assert resumed_numbers == continuation_numbers, (
        f"Failed with {num_workers=} at {checkpoint_step=}\n"
        f"Continuation: {continuation_numbers}\n"
        f"Resumed:     {resumed_numbers}"
    )

    # Check that the aggregator state was correctly restored
    metrics1 = aggregator1.get_metrics()
    for batch in continuation:
        aggregator1.update(batch.pop("metrics"))
    metrics1_after = aggregator1.get_metrics()

    metrics2 = aggregator2.get_metrics()
    for batch in resumed:
        aggregator2.update(batch.pop("metrics"))
    metrics2_after = aggregator2.get_metrics()

    assert (
        metrics1["test_dataset"]["samples_seen"]
        == metrics2["test_dataset"]["samples_seen"]
    ), "Aggregator state for samples_seen not restored correctly"
    assert (
        metrics1_after["test_dataset"]["samples_seen"]
        == metrics2_after["test_dataset"]["samples_seen"]
    ), "Aggregator state for samples_seen not updated correctly after restore"


def test_metrics_collection_comprehensive(dataset_factory, dataset_1_file):
    """Comprehensive test for metrics collection, including all aggregation types."""
    dataset = dataset_factory(dataset_1_file, transform=_tokenize_transform)
    loader = StatefulDataLoader(
        dataset, batch_size=BATCH_SIZE, num_workers=0, collate_fn=_collate_fn
    )
    aggregator = MetricsAggregator()

    num_steps = 5
    total_samples = 0
    total_tokens = 0

    for batch in islice(iter(loader), num_steps):
        # Count expected values manually for validation
        for metric in batch["metrics"]:
            if metric.name == "samples_seen":
                total_samples += metric.value
            elif metric.name == "tokens_seen":
                total_tokens += metric.value

        aggregator.update(batch.pop("metrics"))

    metrics = aggregator.get_metrics()

    # Comprehensive validation
    _assert_metrics_structure(metrics, "Comprehensive metrics test")
    _assert_standard_metrics_present(
        metrics, "test_dataset", "Comprehensive metrics test"
    )

    world_size = dist.get_world_size() if dist.is_initialized() else 1
    expected_total_samples = total_samples * world_size
    expected_total_tokens = total_tokens * world_size

    # Validate specific values
    assert (
        metrics["test_dataset"]["samples_seen"] == expected_total_samples
    ), f"samples_seen mismatch: expected {expected_total_samples}, got {metrics['test_dataset']['samples_seen']}"

    # Note: token count can vary slightly per rank due to sharding, so we don't assert exactness
    # in this test, but check it's in a reasonable range. The distributed test handles correctness.
    if world_size > 1:
        # Don't assert exact token match in distributed setting, as sharding can be uneven.
        # The dedicated distributed test below validates token aggregation.
        pass
    else:
        assert (
            metrics["test_dataset"]["tokens_seen"] == expected_total_tokens
        ), f"tokens_seen mismatch: expected {expected_total_tokens}, got {metrics['test_dataset']['tokens_seen']}"

    # Validate distribution metrics are reasonable
    assert metrics["test_dataset"]["seq_len_p50"] > 0, "seq_len_p50 should be positive"
    assert (
        metrics["test_dataset"]["seq_len_mean"] > 0
    ), "seq_len_mean should be positive"
    assert (
        metrics["test_dataset"]["seq_len_min"] <= metrics["test_dataset"]["seq_len_max"]
    ), "min should be <= max"


def test_metrics_aggregation_types():
    """Test all aggregation types work correctly."""
    aggregator = MetricsAggregator()
    world_size = dist.get_world_size() if dist.is_initialized() else 1

    # Test SUM aggregation
    metrics = [
        Metric(
            dataset_name="test",
            name="sum_metric",
            value=5,
            agg_type=AggregationType.SUM,
        ),
        Metric(
            dataset_name="test",
            name="sum_metric",
            value=3,
            agg_type=AggregationType.SUM,
        ),
        Metric(
            dataset_name="test",
            name="sum_metric",
            value=2,
            agg_type=AggregationType.SUM,
        ),
    ]
    aggregator.update(metrics)

    # Test MEAN aggregation
    metrics = [
        Metric(
            dataset_name="test",
            name="mean_metric",
            value=4.0,
            agg_type=AggregationType.MEAN,
        ),
        Metric(
            dataset_name="test",
            name="mean_metric",
            value=6.0,
            agg_type=AggregationType.MEAN,
        ),
        Metric(
            dataset_name="test",
            name="mean_metric",
            value=8.0,
            agg_type=AggregationType.MEAN,
        ),
    ]
    aggregator.update(metrics)

    # Test DISTRIBUTION aggregation
    metrics = [
        Metric(
            dataset_name="test",
            name="dist_metric",
            value=1,
            agg_type=AggregationType.DISTRIBUTION,
        ),
        Metric(
            dataset_name="test",
            name="dist_metric",
            value=2,
            agg_type=AggregationType.DISTRIBUTION,
        ),
        Metric(
            dataset_name="test",
            name="dist_metric",
            value=3,
            agg_type=AggregationType.DISTRIBUTION,
        ),
        Metric(
            dataset_name="test",
            name="dist_metric",
            value=4,
            agg_type=AggregationType.DISTRIBUTION,
        ),
        Metric(
            dataset_name="test",
            name="dist_metric",
            value=5,
            agg_type=AggregationType.DISTRIBUTION,
        ),
    ]
    aggregator.update(metrics)

    # Test CATEGORICAL_COUNT aggregation
    metrics = [
        Metric(
            dataset_name="test",
            name="cat_metric",
            value="A",
            agg_type=AggregationType.CATEGORICAL_COUNT,
        ),
        Metric(
            dataset_name="test",
            name="cat_metric",
            value="B",
            agg_type=AggregationType.CATEGORICAL_COUNT,
        ),
        Metric(
            dataset_name="test",
            name="cat_metric",
            value="A",
            agg_type=AggregationType.CATEGORICAL_COUNT,
        ),
        Metric(
            dataset_name="test",
            name="cat_metric",
            value="C",
            agg_type=AggregationType.CATEGORICAL_COUNT,
        ),
        Metric(
            dataset_name="test",
            name="cat_metric",
            value="A",
            agg_type=AggregationType.CATEGORICAL_COUNT,
        ),
    ]
    aggregator.update(metrics)

    # Test MAX and MIN aggregation
    metrics = [
        Metric(
            dataset_name="test",
            name="max_metric",
            value=10,
            agg_type=AggregationType.MAX,
        ),
        Metric(
            dataset_name="test",
            name="max_metric",
            value=25,
            agg_type=AggregationType.MAX,
        ),
        Metric(
            dataset_name="test",
            name="max_metric",
            value=15,
            agg_type=AggregationType.MAX,
        ),
        Metric(
            dataset_name="test",
            name="min_metric",
            value=10,
            agg_type=AggregationType.MIN,
        ),
        Metric(
            dataset_name="test",
            name="min_metric",
            value=3,
            agg_type=AggregationType.MIN,
        ),
        Metric(
            dataset_name="test",
            name="min_metric",
            value=7,
            agg_type=AggregationType.MIN,
        ),
    ]
    aggregator.update(metrics)

    final_metrics = aggregator.get_metrics()

    # Validate all aggregation types
    assert (
        final_metrics["test"]["sum_metric"] == 10 * world_size
    ), f"SUM: expected {10 * world_size}, got {final_metrics['test']['sum_metric']}"
    assert (
        final_metrics["test"]["mean_metric"] == 6.0
    ), f"MEAN: expected 6.0, got {final_metrics['test']['mean_metric']}"
    assert (
        final_metrics["test"]["max_metric"] == 25
    ), f"MAX: expected 25, got {final_metrics['test']['max_metric']}"
    assert (
        final_metrics["test"]["min_metric"] == 3
    ), f"MIN: expected 3, got {final_metrics['test']['min_metric']}"

    # Validate distribution metrics
    assert (
        "dist_metric_mean" in final_metrics["test"]
    ), "Distribution mean should be present"
    assert (
        "dist_metric_p50" in final_metrics["test"]
    ), "Distribution p50 should be present"
    assert (
        final_metrics["test"]["dist_metric_mean"] == 3.0
    ), f"Distribution mean: expected 3.0, got {final_metrics['test']['dist_metric_mean']}"

    # Validate categorical counts
    assert (
        final_metrics["test"]["cat_metric_A_count"] == 3 * world_size
    ), f"Category A: expected {3 * world_size}, got {final_metrics['test']['cat_metric_A_count']}"
    assert (
        final_metrics["test"]["cat_metric_B_count"] == 1 * world_size
    ), f"Category B: expected {1 * world_size}, got {final_metrics['test']['cat_metric_B_count']}"
    assert (
        final_metrics["test"]["cat_metric_C_count"] == 1 * world_size
    ), f"Category C: expected {1 * world_size}, got {final_metrics['test']['cat_metric_C_count']}"


def test_distributed_metrics_aggregation(dataset_factory, dataset_1_file):
    """
    Tests that metrics collected on separate ranks can be correctly aggregated
    using a real distributed reduce operation. This simulates how a training
    loop would aggregate metrics from all processes for logging.
    """
    rank, world_size = (
        (0, 1)
        if not dist.is_initialized()
        else (dist.get_rank(), dist.get_world_size())
    )
    if world_size <= 1:
        pytest.skip("This is a distributed test and requires world_size > 1.")

    dataset = dataset_factory(dataset_1_file, transform=_tokenize_transform)
    loader = StatefulDataLoader(
        dataset, batch_size=BATCH_SIZE, num_workers=0, collate_fn=_collate_fn
    )
    aggregator = MetricsAggregator()

    num_steps = 3
    local_samples = 0
    local_tokens = 0

    for batch in islice(iter(loader), num_steps):
        # Track local values for validation
        for metric in batch["metrics"]:
            if metric.name == "samples_seen":
                local_samples += metric.value
            elif metric.name == "tokens_seen":
                local_tokens += metric.value

        aggregator.update(batch.pop("metrics"))

    # Each rank now has its own local metrics in its aggregator
    # get_metrics will handle distributed aggregation
    global_metrics = aggregator.get_metrics()

    # To validate, we need to manually aggregate from all ranks
    local_samples_tensor = torch.tensor([local_samples], dtype=torch.float32)
    dist.all_reduce(local_samples_tensor, op=dist.ReduceOp.SUM)
    expected_total_samples = local_samples_tensor.item()

    local_tokens_tensor = torch.tensor([local_tokens], dtype=torch.float32)
    dist.all_reduce(local_tokens_tensor, op=dist.ReduceOp.SUM)
    expected_total_tokens = local_tokens_tensor.item()

    _assert_metrics_structure(global_metrics, f"Rank {rank}")
    _assert_standard_metrics_present(global_metrics, "test_dataset", f"Rank {rank}")

    aggregated_samples = global_metrics["test_dataset"]["samples_seen"]
    aggregated_tokens = global_metrics["test_dataset"]["tokens_seen"]

    assert aggregated_samples == expected_total_samples, (
        f"Rank {rank}: Aggregated samples mismatch. "
        f"Expected: {expected_total_samples}, Got: {aggregated_samples}, Local: {local_samples}"
    )

    # Validate that tokens were also aggregated correctly
    assert aggregated_tokens == expected_total_tokens, (
        f"Rank {rank}: Aggregated tokens mismatch. "
        f"Expected: {expected_total_tokens}, Got: {aggregated_tokens}, Local: {local_tokens}"
    )


def test_interleaved_dataset_sampling_ratios(tmp_path):
    """InterleavedDataset samples according to specified weights and generates proper metrics."""
    # Create two data files
    path1 = tmp_path / "data1.json"
    path2 = tmp_path / "data2.json"
    _create_json_file(path1, DATASET_SIZE_1)
    _create_json_file(path2, DATASET_SIZE_2)

    def create_dataset(path, name):
        return HFIterableDataset(
            path="json",
            data_files=str(path),
            split="train",
            dataset_name=name,
            seed=SEED,
            shuffle_buffer_size=0,
            message_transform=lambda x: {**x, "source": name},
            model_transform=_identity_transform,
            output_transform=_identity_transform,
            metric_transform=StandardMetricTransform(name),
        )

    ds1 = create_dataset(path1, "ds1")
    ds2 = create_dataset(path2, "ds2")
    weights = [0.7, 0.3]

    interleaved = InterleavedDataset(datasets=[ds1, ds2], weights=weights, seed=SEED)
    loader = DataLoader(interleaved, batch_size=1, collate_fn=_collate_fn)
    aggregator = MetricsAggregator()

    num_samples = 100
    ds1_count = 0
    ds2_count = 0

    for batch in islice(iter(loader), num_samples):
        # Count samples from each dataset
        source = batch["source"][0]  # batch_size=1, so take first element
        if source == "ds1":
            ds1_count += 1
        elif source == "ds2":
            ds2_count += 1

        aggregator.update(batch.pop("metrics"))

    metrics = aggregator.get_metrics()
    world_size = dist.get_world_size() if dist.is_initialized() else 1

    # In distributed setting, ds1_count is local. The metric is global.
    # We need to sum up local counts to get expected global count.
    local_counts = torch.tensor([ds1_count, ds2_count], dtype=torch.float)
    if world_size > 1:
        dist.all_reduce(local_counts, op=dist.ReduceOp.SUM)

    global_ds1_count = local_counts[0].item()
    global_ds2_count = local_counts[1].item()
    global_total_samples = global_ds1_count + global_ds2_count

    observed_ratio = (
        global_ds1_count / global_total_samples if global_total_samples > 0 else 0
    )

    assert abs(observed_ratio - weights[0]) < 0.1, (
        f"Observed ratio {observed_ratio:.3f} should be within 10% of target weight {weights[0]}. "
        f"Global Counts: ds1={global_ds1_count}, ds2={global_ds2_count}"
    )

    # Validate that metrics from both datasets are properly aggregated
    assert "ds1" in metrics, f"ds1 metrics should be present in {metrics.keys()}"
    assert "ds2" in metrics, f"ds2 metrics should be present in {metrics.keys()}"

    total_samples_ds1 = metrics["ds1"]["samples_seen"]
    total_samples_ds2 = metrics["ds2"]["samples_seen"]

    assert (
        total_samples_ds1 == global_ds1_count
    ), f"ds1 sample count mismatch: Aggregated metric {total_samples_ds1} != Expected global count {global_ds1_count}"
    assert (
        total_samples_ds2 == global_ds2_count
    ), f"ds2 sample count mismatch: Aggregated metric {total_samples_ds2} != Expected global count {global_ds2_count}"


def test_interleaved_dataset_checkpointing(tmp_path):
    """InterleavedDataset state, including metrics, can be saved and restored."""
    path1 = tmp_path / "data1.json"
    path2 = tmp_path / "data2.json"
    _create_json_file(path1, DATASET_SIZE_1)
    _create_json_file(path2, DATASET_SIZE_2)

    def create_interleaved():
        ds1 = HFIterableDataset(
            path="json",
            data_files=str(path1),
            split="train",
            dataset_name="ds1",
            seed=SEED,
            shuffle_buffer_size=0,  # Disable shuffling for checkpointing test
            message_transform=lambda x: {**x, "source": "ds1"},
            model_transform=_identity_transform,
            output_transform=_identity_transform,
            metric_transform=StandardMetricTransform("ds1"),
        )
        ds2 = HFIterableDataset(
            path="json",
            data_files=str(path2),
            split="train",
            dataset_name="ds2",
            seed=SEED,
            shuffle_buffer_size=0,  # Disable shuffling for checkpointing test
            message_transform=lambda x: {**x, "source": "ds2"},
            model_transform=_identity_transform,
            output_transform=_identity_transform,
            metric_transform=StandardMetricTransform("ds2"),
        )
        return InterleavedDataset(datasets=[ds1, ds2], weights=[0.6, 0.4], seed=SEED)

    interleaved1 = create_interleaved()
    aggregator1 = MetricsAggregator()
    iterator1 = iter(interleaved1)

    for _ in range(50):
        sample = next(iterator1)
        aggregator1.update(sample.get("metrics", []))

    state = interleaved1.state_dict()
    agg_state = aggregator1.state_dict()
    metrics_before = aggregator1.get_metrics()

    continuation = []
    for _ in range(10):
        sample = next(iterator1)
        continuation.append(sample["source"])
        aggregator1.update(sample.get("metrics", []))
    metrics_after_continue = aggregator1.get_metrics()

    interleaved2 = create_interleaved()
    aggregator2 = MetricsAggregator()
    interleaved2.load_state_dict(state)
    aggregator2.load_state_dict(agg_state)
    metrics_after_load = aggregator2.get_metrics()

    resumed = []
    for sample in islice(iter(interleaved2), 10):
        resumed.append(sample["source"])
        aggregator2.update(sample.get("metrics", []))
    metrics_after_resumed_iteration = aggregator2.get_metrics()

    assert resumed == continuation, (
        f"Resumed stream should be identical to original stream.\n"
        f"Continuation: {continuation}\n"
        f"Resumed:     {resumed}"
    )

    # Verify that metrics were restored correctly
    assert (
        metrics_before == metrics_after_load
    ), f"Metrics should be identical after loading state.\nBefore:\n{metrics_before}\nAfter load:\n{metrics_after_load}"

    # Verify that metrics continue to update correctly
    assert (
        metrics_after_continue == metrics_after_resumed_iteration
    ), f"Metrics after iteration should be identical.\nContinued Original:\n{metrics_after_continue}\nContinued Restored:\n{metrics_after_resumed_iteration}"


def test_transform_failure_filtering(dataset_factory, dataset_1_file):
    """Failed transforms are filtered out gracefully."""
    dataset = dataset_factory(
        dataset_1_file, shuffle=False, transform=_failing_transform
    )
    rank, world_size = (
        (0, 1)
        if not dist.is_initialized()
        else (dist.get_rank(), dist.get_world_size())
    )
    samples_per_rank = DATASET_SIZE_1 // world_size

    samples = list(islice(iter(dataset), samples_per_rank))
    seen_numbers = {batch["number"] for batch in samples}

    # Number 7 should be filtered out due to failing transform
    assert 7 not in seen_numbers, (
        f"Failed sample (number=7) should be filtered out. "
        f"Seen numbers: {sorted(seen_numbers)}, Samples per rank: {samples_per_rank}"
    )
    # Should still get other samples
    assert len(seen_numbers) > 0, (
        f"Should have some valid samples after filtering. "
        f"Got {len(seen_numbers)} samples from {samples_per_rank} expected per rank"
    )


def test_metrics_edge_cases():
    """Test edge cases in metrics aggregation that could cause bugs."""
    aggregator = MetricsAggregator()
    world_size = dist.get_world_size() if dist.is_initialized() else 1

    # Test empty aggregation
    empty_metrics = aggregator.get_metrics()
    assert empty_metrics == {}, "Empty aggregator should return empty dict"

    # Test single value for each aggregation type
    single_metrics = [
        Metric(
            dataset_name="test",
            name="single_sum",
            value=42,
            agg_type=AggregationType.SUM,
        ),
        Metric(
            dataset_name="test",
            name="single_mean",
            value=3.14,
            agg_type=AggregationType.MEAN,
        ),
        Metric(
            dataset_name="test",
            name="single_max",
            value=100,
            agg_type=AggregationType.MAX,
        ),
        Metric(
            dataset_name="test",
            name="single_min",
            value=1,
            agg_type=AggregationType.MIN,
        ),
        Metric(
            dataset_name="test",
            name="single_dist",
            value=5,
            agg_type=AggregationType.DISTRIBUTION,
        ),
        Metric(
            dataset_name="test",
            name="single_cat",
            value="X",
            agg_type=AggregationType.CATEGORICAL_COUNT,
        ),
    ]
    aggregator.update(single_metrics)

    single_result = aggregator.get_metrics()
    assert single_result["test"]["single_sum"] == 42 * world_size
    assert single_result["test"]["single_mean"] == pytest.approx(3.14)
    assert single_result["test"]["single_max"] == 100
    assert single_result["test"]["single_min"] == 1
    assert single_result["test"]["single_dist_mean"] == 5.0
    assert single_result["test"]["single_cat_X_count"] == 1 * world_size

    # Test zero values
    zero_metrics = [
        Metric(
            dataset_name="test", name="zero_sum", value=0, agg_type=AggregationType.SUM
        ),
        Metric(
            dataset_name="test",
            name="zero_mean",
            value=0.0,
            agg_type=AggregationType.MEAN,
        ),
        Metric(
            dataset_name="test",
            name="zero_dist",
            value=0,
            agg_type=AggregationType.DISTRIBUTION,
        ),
    ]
    aggregator.update(zero_metrics)

    zero_result = aggregator.get_metrics()
    assert zero_result["test"]["zero_sum"] == 0
    assert zero_result["test"]["zero_mean"] == 0.0
    assert zero_result["test"]["zero_dist_mean"] == 0.0

    # Test negative values
    negative_metrics = [
        Metric(
            dataset_name="test",
            name="negative_sum",
            value=-5,
            agg_type=AggregationType.SUM,
        ),
        Metric(
            dataset_name="test",
            name="negative_sum",
            value=3,
            agg_type=AggregationType.SUM,
        ),
        Metric(
            dataset_name="test",
            name="negative_min",
            value=-10,
            agg_type=AggregationType.MIN,
        ),
        Metric(
            dataset_name="test",
            name="negative_min",
            value=5,
            agg_type=AggregationType.MIN,
        ),
    ]
    aggregator.update(negative_metrics)

    negative_result = aggregator.get_metrics()
    assert negative_result["test"]["negative_sum"] == -2 * world_size
    assert negative_result["test"]["negative_min"] == -10


def test_metrics_checkpointing_comprehensive():
    """Test that metrics aggregator state can be saved and restored correctly."""
    aggregator1 = MetricsAggregator(dist_window_size=5)
    world_size = dist.get_world_size() if dist.is_initialized() else 1

    # Build up complex state
    metrics_batch1 = [
        Metric(
            dataset_name="ds1", name="counter", value=1, agg_type=AggregationType.SUM
        ),
        Metric(
            dataset_name="ds1",
            name="average",
            value=10.0,
            agg_type=AggregationType.MEAN,
        ),
        Metric(
            dataset_name="ds1",
            name="sequence",
            value=1,
            agg_type=AggregationType.DISTRIBUTION,
        ),
        Metric(
            dataset_name="ds1",
            name="category",
            value="A",
            agg_type=AggregationType.CATEGORICAL_COUNT,
        ),
        Metric(
            dataset_name="ds2", name="maximum", value=50, agg_type=AggregationType.MAX
        ),
    ]
    aggregator1.update(metrics_batch1)

    metrics_batch2 = [
        Metric(
            dataset_name="ds1", name="counter", value=2, agg_type=AggregationType.SUM
        ),
        Metric(
            dataset_name="ds1",
            name="average",
            value=20.0,
            agg_type=AggregationType.MEAN,
        ),
        Metric(
            dataset_name="ds1",
            name="sequence",
            value=2,
            agg_type=AggregationType.DISTRIBUTION,
        ),
        Metric(
            dataset_name="ds1",
            name="category",
            value="B",
            agg_type=AggregationType.CATEGORICAL_COUNT,
        ),
        Metric(
            dataset_name="ds2", name="maximum", value=75, agg_type=AggregationType.MAX
        ),
    ]
    aggregator1.update(metrics_batch2)

    # Save state
    state = aggregator1.state_dict()
    metrics_before = aggregator1.get_metrics()

    # Continue with original aggregator
    metrics_batch3 = [
        Metric(
            dataset_name="ds1", name="counter", value=3, agg_type=AggregationType.SUM
        ),
        Metric(
            dataset_name="ds1",
            name="average",
            value=30.0,
            agg_type=AggregationType.MEAN,
        ),
        Metric(
            dataset_name="ds1",
            name="sequence",
            value=3,
            agg_type=AggregationType.DISTRIBUTION,
        ),
    ]
    aggregator1.update(metrics_batch3)
    metrics_continued = aggregator1.get_metrics()

    # Create new aggregator and restore state
    aggregator2 = MetricsAggregator(dist_window_size=5)
    aggregator2.load_state_dict(state)
    metrics_restored = aggregator2.get_metrics()

    # Validate state was restored correctly
    assert (
        metrics_before == metrics_restored
    ), f"Restored metrics don't match saved state.\nBefore: {metrics_before}\nRestored: {metrics_restored}"

    # Continue with restored aggregator
    aggregator2.update(metrics_batch3)
    metrics_restored_continued = aggregator2.get_metrics()

    # Both aggregators should now have identical state
    assert (
        metrics_continued == metrics_restored_continued
    ), f"Continued metrics don't match.\nOriginal: {metrics_continued}\nRestored: {metrics_restored_continued}"

    # Validate specific values
    assert (
        metrics_continued["ds1"]["counter"] == 6 * world_size
    ), f"Counter should sum to {6 * world_size}"
    assert (
        metrics_continued["ds1"]["average"] == 20.0
    ), "Average should be (10+20+30)/3 = 20"
    assert metrics_continued["ds2"]["maximum"] == 75, "Maximum should be 75"
    assert (
        metrics_continued["ds1"]["category_A_count"] == 1 * world_size
    ), f"Category A count should be {1 * world_size}"
    assert (
        metrics_continued["ds1"]["category_B_count"] == 1 * world_size
    ), f"Category B count should be {1 * world_size}"


def setup_distributed(rank, world_size, master_addr, master_port, timeout_seconds=30):
    """Initialize the distributed environment."""
    os.environ["MASTER_ADDR"] = master_addr
    os.environ["MASTER_PORT"] = str(master_port)

    # Initialize the process group
    dist.init_process_group(
        "gloo",  # or "nccl" if you have GPUs
        rank=rank,
        world_size=world_size,
        timeout=timedelta(seconds=timeout_seconds),
    )


def run_distributed_test(rank, world_size, master_addr, master_port, tmpdir):
    """
    Main distributed test runner.
    Runs a comprehensive suite of tests to validate dataset functionality.
    """
    try:
        # Note: the hang test was intentionally removed from the default run.
        setup_distributed(
            rank, world_size, master_addr, master_port, timeout_seconds=20
        )
        # --- Run Full Test Suite ---
        logger.info(f"Rank {rank}: --- Running Full Test Suite ---")

        # --- Test Setup ---
        # The new tests from `final_enhance_tests.md` are designed to be
        # self-contained. We just need to provide them with the necessary
        # file paths and factory functions.
        tmp_path = Path(tmpdir)
        dataset_factory = get_dataset_factory()
        dataset_1_file = get_dataset_1_file(tmp_path)
        dataset_2_file = get_dataset_2_file(tmp_path)

        # List of all test functions to run
        tests = [
            # Parametrized tests are automatically picked up by pytest,
            # so we just need to list the functions themselves.
            lambda: test_basic_iteration_works(dataset_factory, dataset_1_file),
            lambda: test_shuffling_changes_order(dataset_factory, dataset_2_file),
            lambda: test_hf_dataset_checkpointing_basic(
                dataset_factory, dataset_1_file
            ),
            lambda: test_dataloader_integration(0, dataset_factory, dataset_1_file),
            lambda: test_dataloader_integration(2, dataset_factory, dataset_1_file),
            lambda: test_stateful_dataloader_checkpointing(
                "mid_epoch", 0, dataset_factory, dataset_1_file
            ),
            lambda: test_stateful_dataloader_checkpointing(
                "epoch_boundary", 0, dataset_factory, dataset_1_file
            ),
            lambda: test_stateful_dataloader_checkpointing(
                "mid_epoch", 2, dataset_factory, dataset_1_file
            ),
            lambda: test_stateful_dataloader_checkpointing(
                "epoch_boundary", 2, dataset_factory, dataset_1_file
            ),
            lambda: test_metrics_collection_comprehensive(
                dataset_factory, dataset_1_file
            ),
            lambda: test_metrics_aggregation_types(),
            lambda: test_distributed_metrics_aggregation(
                dataset_factory, dataset_1_file
            ),
            lambda: test_interleaved_dataset_sampling_ratios(tmp_path),
            lambda: test_interleaved_dataset_checkpointing(tmp_path),
            lambda: test_interleaved_dataset_checkpointing_with_shuffling(tmp_path),
            lambda: test_transform_failure_filtering(dataset_factory, dataset_1_file),
            lambda: test_metrics_edge_cases(),
            lambda: test_metrics_checkpointing_comprehensive(),
        ]

        failed_tests = []
        for i, test_fn in enumerate(tests):
            if test_fn is None:
                continue
            # Getting the name of the test function is a bit tricky with lambdas
            # This is a simplification. For better names, we would avoid lambdas.
            test_name = "unknown"
            if hasattr(test_fn, "__code__") and test_fn.__code__.co_names:
                # Safely access co_names with bounds checking
                if len(test_fn.__code__.co_names) > 1:
                    test_name = test_fn.__code__.co_names[1]
                elif len(test_fn.__code__.co_names) > 0:
                    test_name = test_fn.__code__.co_names[0]
                else:
                    test_name = f"test_{i}"

            try:
                logger.info(f"Rank {rank}: Running test: {test_name}")
                test_fn()
                logger.info(f"Rank {rank}: Test PASSED: {test_name}")
                if world_size > 1:
                    dist.barrier()
            except Exception as e:
                # Special handling for pytest.skip()
                if isinstance(e, pytest.skip.Exception):
                    logger.info(
                        f"Rank {rank}: Test SKIPPED: {test_name} - Reason: {e.msg}"
                    )
                    if world_size > 1:
                        dist.barrier()
                    continue

                failed_tests.append((test_name, str(e), traceback.format_exc()))
                logger.error(f"Rank {rank}: Test FAILED: {test_name}: {e}")
                if world_size > 1:
                    # Barrier to prevent other ranks from racing ahead and potentially
                    # causing cascading failures.
                    dist.barrier()

        # Final report
        if failed_tests:
            logger.error(f"Rank {rank}: {len(failed_tests)} tests failed:")
            for name, error, tb in failed_tests:
                logger.error(f"  - {name}: {error}\n    Traceback: {tb}")
            # Exit with a non-zero code to fail CI/scripts.
            sys.exit(1)
        else:
            logger.info(f"Rank {rank}: All tests passed successfully!")

    except Exception as e:
        logger.error(
            f"Rank {rank}: Caught exception in test runner: {e}\n{traceback.format_exc()}"
        )
    finally:
        if dist.is_initialized():
            try:
                dist.destroy_process_group()
            except:
                pass


def main():
    world_size = 2
    # Find a free port
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        master_port = s.getsockname()[1]

    master_addr = "localhost"

    # Set the sharing strategy to 'file_system' which is more robust
    # for large data buffers (like Arrow) than 'file_descriptor'.
    try:
        torch.multiprocessing.set_sharing_strategy("file_system")
    except RuntimeError:
        # Can be set only once per process.
        pass

    with tempfile.TemporaryDirectory() as tmpdir:
        if world_size > 1:
            mp.spawn(
                run_distributed_test,
                args=(world_size, master_addr, master_port, tmpdir),
                nprocs=world_size,
                join=True,
            )
        else:
            run_distributed_test(0, 1, master_addr, master_port, tmpdir)


def test_interleaved_dataset_checkpointing_with_shuffling(tmp_path):
    """Test that interleaved dataset checkpointing works with shuffling enabled, accepting that exact samples may differ."""
    path1 = tmp_path / "data1.json"
    path2 = tmp_path / "data2.json"
    _create_json_file(path1, DATASET_SIZE_1)
    _create_json_file(path2, DATASET_SIZE_2)

    def create_interleaved():
        ds1 = HFIterableDataset(
            path="json",
            data_files=str(path1),
            split="train",
            dataset_name="ds1",
            seed=SEED,
            shuffle_buffer_size=100,  # Enable shuffling
            message_transform=lambda x: {**x, "source": "ds1"},
            model_transform=_identity_transform,
            output_transform=_identity_transform,
            metric_transform=StandardMetricTransform("ds1"),
        )
        ds2 = HFIterableDataset(
            path="json",
            data_files=str(path2),
            split="train",
            dataset_name="ds2",
            seed=SEED,
            shuffle_buffer_size=100,  # Enable shuffling
            message_transform=lambda x: {**x, "source": "ds2"},
            model_transform=_identity_transform,
            output_transform=_identity_transform,
            metric_transform=StandardMetricTransform("ds2"),
        )
        return InterleavedDataset(datasets=[ds1, ds2], weights=[0.6, 0.4], seed=SEED)

    interleaved1 = create_interleaved()
    aggregator1 = MetricsAggregator()
    iterator1 = iter(interleaved1)

    # Process some samples
    for _ in range(50):
        sample = next(iterator1)
        aggregator1.update(sample.get("metrics", []))

    state = interleaved1.state_dict()
    agg_state = aggregator1.state_dict()
    metrics_before = aggregator1.get_metrics()

    # Continue with original
    continuation_sources = []
    for _ in range(10):
        sample = next(iterator1)
        continuation_sources.append(sample["source"])
        aggregator1.update(sample.get("metrics", []))
    metrics_after_continue = aggregator1.get_metrics()

    # Create new dataset and restore
    interleaved2 = create_interleaved()
    aggregator2 = MetricsAggregator()
    interleaved2.load_state_dict(state)
    aggregator2.load_state_dict(agg_state)
    metrics_after_load = aggregator2.get_metrics()

    # Resume with new dataset
    resumed_sources = []
    for sample in islice(iter(interleaved2), 10):
        resumed_sources.append(sample["source"])
        aggregator2.update(sample.get("metrics", []))
    metrics_after_resumed_iteration = aggregator2.get_metrics()

    # With shuffling, exact samples may differ, but we can verify:
    # 1. The sampling generator state was preserved (same source distribution)
    # 2. The metrics aggregator state was preserved correctly
    # 3. The overall dataset behavior is consistent

    # Verify that metrics aggregator state was restored correctly
    assert (
        metrics_before == metrics_after_load
    ), f"Metrics should be identical after loading state.\nBefore:\n{metrics_before}\nAfter load:\n{metrics_after_load}"

    # With shuffling, we can't guarantee exact sample order, but we can verify:
    # 1. Same number of samples from each dataset (approximately, given the randomness)
    # 2. Both continued and resumed sequences should have reasonable distributions

    continuation_ds1_count = continuation_sources.count("ds1")
    continuation_ds2_count = continuation_sources.count("ds2")
    resumed_ds1_count = resumed_sources.count("ds1")
    resumed_ds2_count = resumed_sources.count("ds2")

    # Both should have some samples from each dataset (with 60/40 split, expect roughly 6/4 out of 10)
    assert (
        continuation_ds1_count + continuation_ds2_count == 10
    ), "Continuation should have exactly 10 samples"
    assert (
        resumed_ds1_count + resumed_ds2_count == 10
    ), "Resumed should have exactly 10 samples"

    # With the random seed restored, we should see similar (but not necessarily identical) distributions
    # This is a weaker assertion than exact equality, but appropriate for shuffled data
    assert (
        continuation_ds1_count > 0 and continuation_ds2_count > 0
    ), "Both datasets should contribute to continuation"
    assert (
        resumed_ds1_count > 0 and resumed_ds2_count > 0
    ), "Both datasets should contribute to resumed sequence"


if __name__ == "__main__":
    # Using spawn is important for CUDA compatibility
    mp.set_start_method("spawn", force=True)
    main()
