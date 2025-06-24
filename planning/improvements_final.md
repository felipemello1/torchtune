# Iterable Dataset Final Design Proposal

## Revision Log (as of YYYY-MM-DD)

> **Note to contributors**: To maintain clarity and history, please log significant changes and the reasoning behind them in this section when updating the design document. This follows feedback to keep a clear record of design evolution.

**Latest Update**: This document has been updated based on a second round of feedback to further simplify the design, remove boilerplate, and focus on core functionality.
1.  **Simplified `TuneIterableDataset` Interface**: Removed `skip_seen_samples` and `reshuffle` abstract methods. Resumption logic is now an implicit responsibility of `__iter__` and `state_dict`.
2.  **Refined Metrics Collection**: Integrated sliding window logic directly into `StandardMetricCollector` to prevent memory leaks for statistics like sequence length, while keeping counters for total samples/tokens. The `get_metrics` contract is now `{dataset_name: {"metrics": {...}}}`.
3.  **Corrected `HFIterableDataset` Logic**: Moved lazy Hugging Face operations (`.map`, `.filter`) out of the `__iter__` loop and into the initial setup to prevent re-application on every epoch.
4.  **Robust `InterleavedDataset`**: Added weight normalization. Removed `StopIteration` handling to ensure child datasets are truly infinite and fail fast otherwise.
5.  **Removed Boilerplate**: Replaced the `DatasetNameValidator` class with a simple inline check. Removed illustrative-only classes (`SimpleDataset`, `StatefulDataset`) and complex patterns (`TransformPipeline`, `DatasetInspector`).

Previous updates include:
1. **Refined `HFIterableDataset` API**: Updated to match the desired signature with proper argument names (`message_transform`, `model_transform`, `output_transform`), added missing args (`probability`, `filter_fn`, `filter_kwargs`), and clarified shuffle behavior (shuffle_buffer_size=None/0 means no shuffle).
2. **Fixed Dataset Loading Logic**: Moved `_load_and_shard_dataset` to `__init__` to prevent expensive reloading on every iteration.
3. **Clarified HF Dataset Statefulness**: Added detailed explanation of when HF datasets are/aren't stateful and confirmed `to_iterable_dataset(num_shards)` compatibility with streaming.
4. **Redesigned Metrics Architecture**: Identified and addressed fundamental flaws in the metrics/iteration design. Provided multiple alternative architectures with thorough analysis of pros/cons.
5. **Robust State Management**: Enhanced state_dict/load_state_dict reliability with proper method existence checks and namespace collision prevention.

---

## 1. Overview and Motivation

This document presents a unified design for implementing infinite, stateful, and framework-agnostic iterable datasets in `torchtune`. It synthesizes proposals from multiple sources to create a robust and intuitive system.

### 1.1. Background

The initial implementation of data loading in `torchtune` relied on map-style datasets, which require loading the entire dataset into memory. This approach has several limitations:
- **Memory Inefficiency**: Not suitable for very large datasets that cannot fit in RAM.
- **Distributed Training Issues**: Prone to hangs in distributed settings when data is unevenly distributed across ranks.
- **Lack of Flexibility**: No support for dataset weighting, on-the-fly packing, or streaming from remote sources.

The goal of this redesign is to address these core issues by introducing a new `IterableDataset`-based infrastructure that is scalable, flexible, and robust for distributed training.

### 1.2. Core Requirements

During the design process, several key requirements were identified:

1.  **Infinite Datasets**: To prevent distributed hangs, all datasets must behave as infinite streams of data.
2.  **Stateful Metrics**: Since datasets are infinite, we need robust, stateful logging to monitor training progress (samples, tokens, sequence length statistics) and make informed decisions about `max_steps` and data source weighting.
3.  **Framework Agnostic**: The system must not be tied to a specific library like Hugging Face `datasets`. It should work with any data source that conforms to a simple, stateful iterable protocol.
4.  **Generic Interleaving**: We need a custom, framework-agnostic utility to interleave multiple `TuneIterableDataset` sources based on specified weights.
5.  **Automatic Reshuffling**: Data shuffling should be handled automatically and deterministically for each pass over the data without requiring manual calls like `.set_epoch()` from the training loop.
6.  **Identify Challenges**: Document open questions and challenges for future discussion.

---

## 2. Design Deep Dive: Simplified and Robust Architecture

Based on comprehensive feedback, this section presents a simplified, robust architecture that addresses the critical flaws identified in earlier iterations while maintaining clarity and ease of use.

### 2.1. The `TuneIterableDataset` Base Class

#### Motivation

**Problem: Infinite Datasets by Default**

In distributed training, all ranks must synchronize for collective operations. If a dataset is finite, one rank might exhaust its data while others are still computing, causing the training job to hang. Similarly, when interleaving multiple datasets, shorter datasets will exhaust before longer ones. All datasets must behave as infinite streams to prevent this.

**Problem: Framework-Agnostic Interface**

`torchtune` should not be tightly coupled to a single data-loading library like Hugging Face `datasets`. The core components should operate on a general interface, allowing users to plug in data from any source (Parquet files, WebDataset, custom generators) as long as it conforms to the contract.

**Solution: A Common Abstract Base Class**

The `TuneIterableDataset` abstract base class serves as the framework-agnostic interface that solves these problems.
- It inherits from `torch.utils.data.IterableDataset` and `abc.ABC`.
- Each implementation is responsible for providing an infinite `__iter__` method. The training loop becomes solely responsible for termination (e.g., via `max_steps`).
- It defines a minimal, consistent contract required for all dataset implementations to ensure they are compatible with the training loop, checkpointing, and metric logging systems. The required methods are `get_metrics`, `state_dict`, and `load_state_dict`.
- Any component that processes or combines datasets, such as the interleaver or packer, will be implemented to operate *only* on `TuneIterableDataset` objects.
- We will provide a `HFIterableDataset` as a concrete implementation for convenience, but it will be just one of many possible implementations.

#### Implementation

```python
from abc import ABC, abstractmethod
from typing import Dict, Any, Iterator, Optional, Protocol, List, Callable
import torch
from torch.utils.data import IterableDataset
import collections
import traceback
from dataclasses import dataclass

@dataclass
class DatasetMetrics:
    """
    Standard structure for dataset metrics. Enforces that metrics are numeric
    and separates them from other metadata.
    """
    metrics: Dict[str, float]  # Only numeric values for logging
    metadata: Dict[str, Any] = None  # Optional non-metric data
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
        
        # Validate that all metrics are numeric
        for key, value in self.metrics.items():
            if not isinstance(value, (int, float)):
                raise ValueError(f"Metric '{key}' must be numeric, got {type(value)}")

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
```

### 2.2. Composable and Stateful Metrics

#### Motivation

**Problem: Composable Metrics & Logging**

With infinite datasets, we need robust, stateful metrics to monitor training progress, token usage, and data quality. We need to track how many "epochs" have been completed, sample/token counts, and sequence length distributions, with the ability to see these stats per-source and for final packed batches. The original metrics design was error-prone with unclear responsibilities and double-counting risks.

**Solution: Composable Metric Collectors**

Use a clean separation where each dataset implements its own metrics logic. A `StandardMetricCollector` provides a reusable component with memory-safe sliding window statistics, but its usage is optional. The base class defines the interface, and datasets can implement metrics simply using dictionaries. This pattern, often called **Hierarchical Metrics Collection**, is a best practice. A `MetricCollector` protocol defines the interface for reusable metric components.

#### Implementation

```python
class MetricCollector(Protocol):
    """
    A protocol defining the interface for a reusable metric collection component.
    This allows for different metric collection strategies to be plugged into datasets,
    promoting separation of concerns between data iteration and metric calculation.
    """

    def update_metrics(self, sample: Dict[str, Any]) -> None:
        """Updates the internal metric counters based on a single data sample."""
        ...

    def get_metrics(self) -> Dict[str, Any]:
        """Returns the current state of all collected metrics."""
        ...

    def increment_epoch(self) -> None:
        """
        Signals to the collector that a full pass over the underlying data has
        been completed. This is used to update epoch-based counters.
        """
        ...

    def state_dict(self) -> Dict[str, Any]:
        """Returns the collector's internal state for checkpointing."""
        ...

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Loads the collector's state from a checkpoint."""
        ...

class StandardMetricCollector(MetricCollector):
    """
    Hugging Face dataset implementation with composable metrics.
    This class is a self-contained component for fetching, transforming, shuffling and iterating
    over data from the Hugging Face Hub while tracking key metrics.
    """
    
    def __init__(
        self,
        dataset_name: str,
        seq_len_stat_window_size: int = 1000,
    ):
        self.dataset_name = dataset_name
        self._token_keys = ["tokens", "input_ids"]
        self._metrics = {
            "samples_seen": 0,
            "tokens_seen": 0,
            "epochs_completed": 0,
        }
        self._seq_len_stat_window = collections.deque(
            maxlen=seq_len_stat_window_size
        )

    def update_metrics(self, sample: Dict[str, Any]) -> None:
        """Update sample count and token metrics from the first found token key."""
        self._metrics["samples_seen"] += 1
        
        for token_key in self._token_keys:
            if token_key in sample and sample[token_key] is not None:
                seq_len = len(sample[token_key])
                self._metrics["tokens_seen"] += seq_len
                self._seq_len_stat_window.append(seq_len)
                break
    
    def get_metrics(self) -> Dict[str, Dict[str, Any]]:
        """Return current metrics including sequence length statistics."""
        metrics_snapshot = self._metrics.copy()
        
        if self._seq_len_stat_window:
            window_list = list(self._seq_len_stat_window)
            window_tensor = torch.tensor(window_list, dtype=torch.float32)
            metrics_snapshot["seq_len_p50"] = torch.quantile(window_tensor, 0.5).item()
            metrics_snapshot["seq_len_p95"] = torch.quantile(window_tensor, 0.95).item()
            metrics_snapshot["seq_len_mean"] = window_tensor.mean().item()
            metrics_snapshot["seq_len_window_size"] = len(window_list)
        
        return {self.dataset_name: {"metrics": metrics_snapshot}}

    def increment_epoch(self) -> None:
        """Call when an epoch completes."""
        self._metrics["epochs_completed"] += 1
    
    def state_dict(self) -> Dict[str, Any]:
        """Return metric state for checkpointing."""
        return {
            "metrics": self._metrics.copy(),
            "seq_len_stat_window": list(self._seq_len_stat_window),
        }
    
    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Load metric state from checkpoint."""
        self._metrics.update(state_dict.get("metrics", {}))
        self._seq_len_stat_window.extend(state_dict.get("seq_len_stat_window", []))
```

### 2.3. Hugging Face Dataset Adapter

#### Motivation

**Problem: Automatic Reshuffling**

The training loop should be simple. Manually calling a `.set_epoch()` method on the dataset is brittle and leaks implementation details. A dataset should be self-contained and manage its own shuffling randomness.

**Solution: Self-Contained Shuffling**

This is handled directly and elegantly by the dataset implementation.
- It maintains an internal `_num_epochs` counter, which is persisted in the `state_dict`.
- On every new pass, it increments this counter.
- A new, deterministic seed for the pass is generated via `self._seed + self._num_epochs`.
- This "epoch seed" is used internally by the underlying `datasets.IterableDataset.shuffle()` method.

This design guarantees:
1.  Data is reshuffled on every pass.
2.  The sequence of data across all passes is fully reproducible from the single initial seed and the number of epochs completed.
3.  The training loop is clean, with no responsibility for managing dataset epochs or seeds.

#### Implementation

The `HFIterableDataset` is a concrete implementation of `TuneIterableDataset` for Hugging Face `datasets`. It demonstrates how a concrete implementation can use a `MetricCollector` and manage its own shuffling and state.

```python
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
        shuffle_buffer_size: Optional[int] = 1000,
        seed: int = 42,
        num_shards_per_worker: int = 16,
        dataset_name: Optional[str] = None,
        metric_collector: Optional[MetricCollector] = None,
        filter_fn: Optional[Callable] = None,
        filter_kwargs: Optional[Dict[str, Any]] = None,
        **load_dataset_kwargs,
    ):
        # Auto-generate dataset name if not provided
        if dataset_name is None:
            path = load_dataset_kwargs.get('path', 'unknown')
            split = load_dataset_kwargs.get('split', 'train')
            self._dataset_name = f"{path.replace('/', '_')}_{split}"
        else:
            self._dataset_name = dataset_name
        
        # Store configuration
        self._shuffle_buffer_size = shuffle_buffer_size
        self._seed = seed
        self._transforms = {
            "message": message_transform,
            "model": model_transform,
            "output": output_transform
        }
        
        # Setup composable metrics
        self.metric_collector = metric_collector or StandardMetricCollector(self.dataset_name)
        
        # Internal state for resumption
        self._num_epochs = 0
        
        # Load and setup HF dataset
        self._setup_hf_dataset(
            load_dataset_kwargs, 
            num_shards_per_worker,
            filter_fn, 
            filter_kwargs)
    
    @property
    def dataset_name(self) -> str:
        return self._dataset_name

    def _setup_hf_dataset(self, 
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
        from datasets import load_dataset, split_dataset_by_node
        
        # Distributed setup
        world_size, rank = 1, 0
        if torch.distributed.is_initialized():
            world_size = torch.distributed.get_world_size()
            rank = torch.distributed.get_rank()

        # Load and shard dataset
        num_shards = world_size * num_shards_per_worker
        ds = load_dataset(**load_dataset_kwargs)
        ds = ds.to_iterable_dataset(num_shards=num_shards)
        
        # Apply filtering if specified
        if filter_fn:
            filter_kwargs = filter_kwargs or {}
            ds = ds.filter(filter_fn, **filter_kwargs)
        
        # Distribute across ranks
        if world_size > 1:
            ds = split_dataset_by_node(ds, rank=rank, world_size=world_size)

        # Apply lazy transforms once during setup, not in __iter__
        self._ds = ds.map(self._apply_transforms).filter(
            lambda x: not x.get("__failed_transform__", False)
        )
    
    def __iter__(self) -> Iterator[Dict[str, Any]]:
        """Iterate through the dataset with automatic metrics collection."""
        while True:  # Infinite iteration
            epoch_seed = self._seed + self._num_epochs
            
            epoch_ds = self._ds
            if self._shuffle_buffer_size and self._shuffle_buffer_size > 0:
                epoch_ds = epoch_ds.shuffle(seed=epoch_seed, buffer_size=self._shuffle_buffer_size)
            
            iterator = iter(epoch_ds)
            
            # NOTE: We do not need to manually skip samples for resumption.
            # Hugging Face's `to_iterable_dataset` is stateful and handles
            # this automatically. `state_dict()` on the dataset object will
            # capture the sharding and sample progress, and `load_state_dict()`
            # will resume from that exact point.

            for sample in iterator:
                self.metric_collector.update_metrics(sample)
                yield sample
            
            # Epoch complete
            self._num_epochs += 1
            self.metric_collector.increment_epoch()
    
    def _apply_transforms(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """Apply the transform pipeline with simple error handling."""
        try:
            sample = self._transforms["message"](sample)
            sample = self._transforms["model"](sample)
            sample = self._transforms["output"](sample)
            return sample
        except Exception:
            return {
                "__failed_transform__": True,
                "error_traceback": traceback.format_exc(),
                "original_sample": sample
            }
    
    def get_metrics(self) -> Dict[str, Any]:
        """Return metrics from the collector."""
        return self.metric_collector.get_metrics()
    
    def state_dict(self) -> Dict[str, Any]:
        """
        Return state for checkpointing, including the state of the underlying
        Hugging Face IterableDataset to ensure exact resumption.
        """

        hf_dataset_state = self._ds.state_dict()

        state = {
            "num_epochs": self._num_epochs,
            "seed": self._seed,
            "metric_state": self.metric_collector.state_dict(),
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
        self.metric_collector.load_state_dict(own_state["metric_state"])
        hf_dataset_state = own_state.get("hf_dataset_state")
        self._ds.load_state_dict(hf_dataset_state)
```

### 2.4. Composite Datasets and State Management

#### Motivation

**Problem: Generic Interleaving and Packing**

Given the need for a framework-agnostic interface, we need custom utilities that can combine `TuneIterableDataset`s, such as an interleaver that mixes datasets with given weights, or a packer that combines multiple samples into a single sequence.

**Problem: Simplified and Robust State Management**

Complex inheritance-based state management is fragile. Composite datasets like an interleaver or packer need a simple and direct way to manage their own state and the state of their underlying child datasets.

**Solution: Direct, Namespaced State Management**

Each dataset implements its own state management. Composite datasets iterate through their children and collect their states directly, enforcing unique dataset names to prevent key collisions in the final checkpoint. This avoids fragile `super()` calls and makes debugging checkpoints easier.

#### Implementation

The `InterleavedDataset` and `IterablePackedDataset` are examples of composite datasets that operate on other `TuneIterableDataset` instances.

```python
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
        dataset_name: str = "interleaved_dataset"
    ):
        # Validate unique dataset names upfront - fail fast with clear error
        names = [ds.dataset_name for ds in datasets]
        if len(names) != len(set(names)):
            duplicates = [name for name, count in collections.Counter(names).items() if count > 1]
            raise ValueError(
                f"Duplicate dataset names detected: {duplicates}. All {names=}"
                f"Please provide a unique 'dataset_name' for each dataset in the interleaved list."
            )
        
        self._dataset_name = dataset_name
        self._datasets = datasets
        
        # Normalize weights to sum to 1
        total_weight = sum(weights)
        self._weights = torch.tensor([w / total_weight for w in weights], dtype=torch.float)
        if total_weight != 1.0:
            logger.warning(f"Interleaved dataset normalized weights to sum to 1.0. Previous {weights=}, new {self._weights.tolist()}")

        self._sampling_generator = torch.Generator().manual_seed(seed)
        
        # Simple metrics for the interleaver itself
        self._metrics = {"interleaved_samples_seen": 0}
    
    @property
    def dataset_name(self) -> str:
        return self._dataset_name

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        """Interleave samples from child datasets. Fails fast if a child is not infinite."""
        child_iters = [iter(ds) for ds in self._datasets]
        
        while True:
            # Sample which dataset to use
            ds_idx = torch.multinomial(
                self._weights, 1, replacement=True, generator=self._sampling_generator
            ).item()
            
            try:
                sample = next(child_iters[ds_idx])
                self._metrics["interleaved_samples_seen"] += 1
                yield sample
            except StopIteration:
                logger.warning(f"Child dataset {self._datasets[ds_idx].dataset_name} is exhausted. Expected an infinite dataset.")
                child_iters[ds_idx] = iter(self._datasets[ds_idx])
                sample = next(child_iters[ds_idx])
                self._metrics["interleaved_samples_seen"] += 1
                yield sample
    
    def get_metrics(self) -> Dict[str, Any]:
        """Collect metrics from self and all children."""
        metrics = {
            self.dataset_name: {
                "metrics": self._metrics.copy()
            }
        }

        # Collect child metrics
        # e.g.
        # {
        # {"ds1": {"metrics": {"samples_seen": 100}}}
        # {"ds2": {"metrics": {"samples_seen": 200}}}
        # {"this_dataset": {"metrics": {"samples_seen": 300}}}
        # }
        for ds in self._datasets:
            metrics.update(ds.get_metrics())
        
        return metrics
    
    def state_dict(self) -> Dict[str, Any]:
        """Save state for the interleaver and its children."""
        child_states = {}
        for ds in self._datasets:
            child_states.update(ds.state_dict())

        state = {
            "metrics": self._metrics.copy(),
            "sampling_generator_state": self._sampling_generator.get_state(),
            "child_states": child_states
        }
        
        return {self.dataset_name: state}
    
    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Load state for the interleaver and its children."""
        own_state = state_dict[self.dataset_name]
        
        self._metrics.update(own_state["metrics"])
        self._sampling_generator.set_state(own_state["sampling_generator_state"])
        
        child_states = own_state["child_states"]
        for ds in self._datasets:
            if ds.dataset_name in child_states:
                ds.load_state_dict(child_states)

class IterablePackedDataset(TuneIterableDataset):
    """
    Takes an iterable dataset and packs its samples together.
    NOTE: This is a simplified design sketch. For more details on the packing
    logic, see `planning/packed_dataset.py`.
    """
    def __init__(self, dataset: TuneIterableDataset, packing_strategy, **kwargs):
        self._child_dataset = dataset
        self._strategy = packing_strategy
        self._dataset_name = f"packed_{self._child_dataset.dataset_name}"
        # ... other packing-related initializations ...
        self._metrics = {
            "num_packed_samples": 0,
            "packing_efficiency": 0.0,
        }

    @property
    def dataset_name(self) -> str:
        return self._dataset_name

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        # This would contain the core packing logic:
        # 1. Create an iterator from the child dataset.
        # 2. Fill a buffer of samples.
        # 3. Create packs from the buffer until a stopping condition.
        # 4. Yield the pack.
        # 5. Update self._metrics on each yielded pack.
        pass

    def get_metrics(self) -> Dict[str, Any]:
        """Hierarchically collects metrics from itself and its child."""
        # First, get our own metrics
        own_metrics = {self.dataset_name: {"metrics": self._metrics.copy()}}
        
        # Then, recursively get metrics from the child and merge
        child_metrics = self._child_dataset.get_metrics()
        
        # Structure can be flattened or nested. Here's a nested example:
        own_metrics[self.dataset_name]["source_dataset_metrics"] = child_metrics
        return own_metrics

    def state_dict(self) -> Dict[str, Any]:
        """Save packer's state and hierarchically save child's state."""
        state = {
            "metrics": self._metrics.copy(),
            "child_state": self._child_dataset.state_dict(),
            # ... other packer-specific state ...
        }
        return {self.dataset_name: state}

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Load packer's state and hierarchically load child's state."""
        own_state = state_dict[self.dataset_name]
        self._metrics.update(own_state["metrics"])
        self._child_dataset.load_state_dict(own_state["child_state"])
        # ... load other packer-specific state ...
```

### 2.5. Complete Integration Example

This example shows how the simplified components can be instantiated and used in a training loop, including efficient distributed metric logging.

```python
# In a recipe's setup_data function
def setup_data(
    self,
    cfg_dataset: ConfigDict|List[ConfigDict],
    cfg_dataloader: ConfigDict,
    cfg_packing: Optional[ConfigDict] = None,
    multidataset_stopping_strategy: str = "first_exhausted",
    dataloader_state_dict: Optional[Dict] = None,
) -> StatefulDataLoader:
    """
    All data related setup happens here. If a state_dict is provided (meaning we are resuming a training run),
    it is loaded into the dataloader.
    """
    seed = self.seed
    pad_id = self._tokenizer.pad_id
    ignore_idx = self._loss_fn.ignore_index
    pad_to_multiple_of = self.parallel_dims.min_seq_len_divisor
    tokenizer = self._tokenizer

    iterable_datasets = []
    weights = []

    # Add dataset to a list just for processing
    if not isinstance(cfg_dataset, list):
        cfg_dataset = [cfg_dataset]
    
    # ---- instantiate datasets ----
    for cfg in cfg_dataset:
        weight = cfg.get("weight", 1.0)
        weights.append(weight)
        
        ds = instantiate(
            seed=self.seed,
            model_transform=model_transform,
            **cfg,
        )

        iterable_datasets.append(ds)

    ds = interleave_datasets(iterable_datasets, weights, seed, multidataset_stopping_strategy)

    # Packing
    if cfg_packing:
        ds = instantiate(
            cfg_packing,
            dataset=ds,
            padding_idx=pad_id,
            ignore_idx=ignore_idx,
        )

    # ---- Instantiate collate_fn ---- 
    collate_fn = dataloader_cfg.pop("collate_fn", None)
    if collate_fn is None:
        collate_fn = (
            "torchtune.data.padded_collate_packed"
            if packing else
            "torchtune.data.padded_collate_sft"
        )

    collate_fn = _get_component_from_path(collate_fn)
    collate_fn = partial(
        collate_fn,
        padding_idx=pad_id,
        ignore_idx=ignore_id,
        pad_to_multiple_of=pad_to_multiple_of
    )

    # ----  Instantiate dataloader ---- 
    # Dropping last avoids shape issues with compile + flex attention
    if "drop_last" not in dataloader_cfg:
        dataloader_cfg["drop_last"] = True

    dataloader = StatefulDataLoader(dataset=ds, collate_fn=collate_fn, **dataloader_cfg)

    if dataloader_state_dict is not None:
        dataloader.load_state_dict(dataloader_state_dict)

    return dataloader

# In the main training recipe
class FullFinetune:
    def train(self):
        # ...
        for step, batch in enumerate(self._dataloader):
            if step >= self._max_steps:
                break
            
            # ... training logic ...
            
            # Log dataset metrics periodically
            if step % self._dataset_metrics_log_freq == 0:
                # get_metrics() returns a flat dict of all datasets
                metrics = self._dataloader.dataset.get_metrics()
                self._log_dataset_metrics(metrics, step)
    
    def _log_dataset_metrics(self, nested_metrics: Dict[str, Any], step: int):
        """
        Flatten, reduce, and log the nested metrics structure in a distributed-safe way.
        """
        # On non-rank-0, do nothing for logging.
        # In distributed settings, we gather metrics from all ranks to rank 0.
        rank = 0
        world_size = 1
        if torch.distributed.is_initialized():
            rank = torch.distributed.get_rank()
            world_size = torch.distributed.get_world_size()

        flat_metrics = {}
        # First, flatten the metric structure for easier processing
        for dataset_name, data in nested_metrics.items():
            if "metrics" in data and isinstance(data["metrics"], dict):
                for metric_name, value in data["metrics"].items():
                    if isinstance(value, (int, float)):
                        key = f"dataset/{dataset_name}/{metric_name}"
                        flat_metrics[key] = value
        
        #TODO: double check if this is the best way of doing this
        # even for huggingface, maybe we can log the dict directly instead of making N calls.
        if world_size > 1:
            # Create a tensor of all metric values
            metric_values = torch.tensor(
                list(flat_metrics.values()), device=f"cuda:{rank}"
            )
            # Reduce (sum) all metrics across all ranks.
            # This is efficient as it's a single collective operation.
            torch.distributed.all_reduce(metric_values, op=torch.distributed.ReduceOp.SUM)
            
            # Only rank 0 will log the final summed metrics
            if rank == 0:
                reduced_values = metric_values.cpu().tolist()
                for i, key in enumerate(flat_metrics.keys()):
                    self.logger.log(key, reduced_values[i], step)
        else:
            # For single-GPU or CPU, just log directly
            for key, value in flat_metrics.items():
                self.logger.log(key, value, step)


    def save_checkpoint(self, checkpoint_path):
        # ...
        checkpoint["dataloader_state_dict"] = self._dataloader.state_dict()
        # ...
    
    def load_checkpoint(self, checkpoint_path):
        # ...
        if "dataloader_state_dict" in checkpoint:
            self._dataloader.load_state_dict(checkpoint["dataloader_state_dict"])
        # ...
```

This simplified architecture provides:

1. **Clear Responsibilities**: Each dataset handles its own iteration, metrics, and state.
2. **No Inheritance Complexity**: Direct implementation without fragile `super()` calls.
3. **Safe Composition**: Explicit validation prevents silent naming errors.
4. **Consistent Interface**: All datasets return data in a predictable format for logging.
5. **Composable Metrics**: Optional metric collectors for complex cases like HF datasets.
6. **Simple Debugging**: State dicts include dataset names for easy identification.
7. **Memory Safety**: `StandardMetricCollector` uses sliding windows for statistics to prevent unbounded memory growth.

---

## 3. Next Steps
The next step is to implement the components described in this document, with a clear plan for integration:
- **Adopt `StatefulDataLoader`**: Adopt `torchdata.StatefulDataLoader` as a dependency.
- Implement the `TuneIterableDataset` abstract base class as designed.
- Implement a concrete `HFIterableDataset` implementation.
- Implement the `InterleavedDataset` as designed.
- Implement a `PackingWrapper` dataset.
- Add comprehensive testing for distributed scenarios and checkpointing, focusing on reproducibility after resumption.
- Update the recipe's `setup_data` utility to construct dataset pipelines using these new components with a generic factory.
- Integrate metric logging into the training loop, which will query the dataloader's `dataset.get_metrics()` method at regular intervals.
