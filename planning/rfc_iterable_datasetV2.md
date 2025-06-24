### Core Issues

1. No support for iterative dataset:
   - Dataset has to be fully loaded in memory
   - With map-style, no control over multi-sample operations (e.g. packing or skipping)
   - Map-style is slower
   - No support for streaming

2. No support for weighted dataset:
   - We have it in a single newly added dev recipe/config, but API needs polishing
   - We also support ConcatDataset, but it's map style and there is no weighting

3. No support for on-the-fly data packing:
   - It's done before training, taking a long time for large datasets

### Proposal

#### Config Example (config.yaml)

```yaml
###########
# Tokenizer
###########
tokenizer:
  _component_: torchtune.models.llama3_2_vision.llama3_2_vision_transform
  path: /tmp/Llama-3.2-11B-Vision-Instruct/original/tokenizer.model
  image_size: 560
  max_seq_len: 8192

##########
# Dataloader
# Consolidate all dataloader args here (currently scattered)
##########
dataloader:
  batch_size: 4
  num_workers: 4
  pin_memory: true
  collate_fn: torchtune.data.padded_collate

#########
# Dataset Options
#########

# Option 1: Direct Class Usage (current SFTDataset approach)
dataset:
  - _component_: alpaca_iterable_dataset
    path: "tatsu-lab/alpaca"
    split: "train"
    message_transform:
        _component_: torchtune.datasets.alpaca_message_transform
        masking_strategy: "output_only"
        column_map:
            input: "prompt"
            output: "response"
            system_prompt: "foo"
    filter_fn: 
        _component_: torchtune.datasets.filter_fn_even_indices
    filter_kwargs:      
        with_indices: True
    weight: 0.8
  - _component_: sft_iterable_dataset
    path: "tatsu-lab/gsm8k"
    split: "train"
    message_transform:
        _component_: torchtune.datasets.gsm8k_message_transform
        masking_strategy: "output_only"
        column_map:
            input: "prompt"
            output: "response"
            system_prompt: "bar"
    weight: 0.2

#########
# Dataset Setup Arguments (not dataset specific)
#########
packing:
    _component_: torchtune.datasets.packing.SFTPacking
    max_seq_len: ${tokenizer.max_seq_len}
multidataset_stopping_strategy: "first_exhausted"  # or "all_exhausted"
```

#### Iterable Dataset Implementation
This is shared for all datasets and recipes (SFT, DPO, etc). Differences are in the transforms.
Location: torchtune/datasets/hf_iterable_dataset.py


```python

def alpaca_iterable_dataset(train_on_input, column_map, *args, **kwargs):
    message_transform = AlpacaToMessages(
        train_on_input=train_on_input, column_map=column_map
    )
    return sft_iterable_dataset(
        message_transform=message_transform,
        *args,
        **kwargs
    )

def sft_iterable_dataset(tokenizer, *args, **kwargs)
    output_transform = SFTOutputTransform(tokenizer.ignore_idx)
    return HfIterableDataset(  
        model_transform=tokenizer,
        output_transform=output_transform,
        *args,
        **kwargs
    )

def _filter_failed_transforms(example: dict) -> bool:
    """Filter function to remove samples that failed during the transform phase."""
    return not example.get("__failed_transform__", False)


class HfIterableDataset(TorchIterableDataset, Stateful):
    def __init__(
        self,
        *,
        message_transform: Transform,
        model_transform: Transform,
        output_transform: Transform,
        shuffle_buffer_size: Optional[int] = 1000,
        seed: Optional[int] = 42,
        num_shards_per_worker: int = 16,
        probability: float = 1.0,
        filter_fn: Optional[Callable] = None,
        filter_kwargs: Optional[Dict] = None,
        max_error_count: int = 10,
        **load_dataset_kwargs,
    ):
        self.message_transform = message_transform
        self.model_transform = model_transform
        self.output_transform = output_transform
        self.weight = weight
        self.error_count = 0
        self.max_error_count = max_error_count
        self.enable_repeat = enable_repeat

        world_size, rank = 1, 0
        if torch.distributed.is_initialized():
            world_size = torch.distributed.get_world_size()
            rank = torch.distributed.get_rank()

        num_shards = world_size * num_shards_per_worker
        ds = load_dataset(**load_dataset_kwargs)
        ds = ds.to_iterable_dataset(num_shards)

        if filter_fn:
            ds = ds.filter(filter_fn, **filter_kwargs)

        ds = ds.map(self._apply_transforms_and_handle_errors)
        ds = ds.filter(_filter_failed_transforms)

        if shuffle_buffer_size and shuffle_buffer_size > 0:
            ds = ds.shuffle(buffer_size=shuffle_buffer_size, seed=seed)

        if world_size > 1:
            ds = split_dataset_by_node(ds, rank=rank, world_size=world_size)

        self.ds = ds

    def _apply_transforms_and_handle_errors(self, sample):
        try:
            sample = self.message_transform(sample)
            sample = self.model_transform(sample)
            sample = self.output_transform(sample)
            return sample
        except Exception:
            self.error_count += 1
            if self.error_count > self.max_error_count:
                raise
            return {"__failed_transform__": True}

    def __iter__(self):
        while True:
            self.error_count = 0
            data_iterator = iter(self.ds)
            for sample in data_iterator:
                yield sample
            if not self.enable_repeat:
                break

    def state_dict(self):
        if not isinstance(self.ds, HFIterableDataset):
            raise TypeError(
                "state_dict() can only be called on a datasets.IterableDataset object"
            )
        state_dict = self.ds.state_dict()
        state_dict["weight"] = self.weight
        return state_dict

    def load_state_dict(self, state_dict):
        if not isinstance(self.ds, HFIterableDataset):
            raise TypeError(
                "load_state_dict() can only be called on a datasets.IterableDataset object"
            )
        self.weight = state_dict.pop("weight")
        self.ds.load_state_dict(state_dict)
```

#### Setup Data
Method in recipes/full_distributed.py or *utility* used in the recipe

```python
from datasets import interleave_datasets, split_dataset_by_node
from torchtune.models.tokenizers import ModelTokenizer
import torch

# TODO: move to torchtune.datasets.utils.py
def interleave_datasets(
    iterable_datasets: List[IterableDataset], 
    weights: List[float], 
    seed: int = 42, 
    multidataset_stopping_strategy: Literal["first_exhausted", "all_exhausted"] = "first_exhausted", 
):

    # Interleave for multidataset
    if len(iterable_datasets) > 1:
        sum_weights = sum(weights)
        if sum_weights != 1:
            probabilities = [w/sum_weights for w in weights]
            logger.info(f"When interleaving datasets, found sum({weights=}) != 1. Normalizing to {probabilities=}")
        ds = interleave_datasets(
            iterable_datasets,
            probabilities=probabilities,
            seed=seed,
            # strategies: https://huggingface.co/docs/datasets/v3.3.2/en/package_reference/main_classes#datasets.interleave_datasets.stopping_strategy
            stopping_strategy=multidataset_stopping_strategy,
        )
    else:
        ds = iterable_datasets[0]

    return ds

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
```

#### Recipe Train Loop

```python
for i, example in enumerate(dataloader):
    if i > max_steps:
        break
    pass
```

### Potential Distributed Training Issues

The use of `IterableDataset` in a distributed setting introduces several complexities that can lead to hangs, crashes, or silent correctness issues. Here are some of the key challenges and considerations:

1.  **Uneven Number of Batches Across Ranks**:
    This is the most common and critical issue. Distributed training strategies like FSDP rely on collective communication operations (e.g., `all_reduce`) where all processes (ranks) must participate. If one rank has more data batches than another, the process with more batches will eventually reach a collective operation while other processes have already exited their training loop. This will cause a hang. This can happen due to:
    - **Unevenly sized shards**: If the dataset is split into shards of different sizes.
    - **Filtering**: A `filter_fn` might discard a different number of samples on each rank, especially if the filtering logic is data-dependent.
    - **Dynamic data sources**: In streaming scenarios, there's no guarantee of an equal number of samples per rank.
    - **`interleave_datasets` with `"all_exhausted"`**: When using `interleave_datasets` with `stopping_strategy="all_exhausted"`, datasets can be exhausted at different times. If the remaining datasets don't have data for all ranks, it can lead to an imbalance.

    **Mitigation**: The `multidataset_stopping_strategy: "first_exhausted"` is a good default to ensure that the training stops as soon as one of the datasets is exhausted, which helps in cases of multiple datasets. For single datasets or unavoidable imbalances, a custom sampler or dataloader wrapper might be needed to either pad the shorter iterators or truncate all to the length of the shortest one. The use of `drop_last=True` in the `DataLoader` is also recommended to avoid issues with partially filled last batches.

2.  **DataLoader Workers and Sharding**:
    - **`num_workers` vs. `num_shards`**: A potential concern is the interaction between the number of `DataLoader` workers and the number of shards. The setup in the RFC is `num_shards = world_size * num_shards_per_worker`. This means each rank gets `num_shards_per_worker` shards. The `DataLoader` on a given rank will then distribute these shards among its `num_workers`. If `num_workers > num_shards_per_worker`, multiple workers will be reading from the same shard. While `datasets` library handles this, it might lead to I/O contention or inefficient data loading.
    - **Worker-level sharding**: PyTorch's `DataLoader` with an `IterableDataset` will give each worker a copy of the dataset object. The `datasets` library `IterableDataset` is aware of this and will use `torch.utils.data.get_worker_info()` to further split its assigned data among the workers on that rank. This is a complex interaction that needs to be working correctly to avoid data duplication or data loss.

3.  **Streaming from Remote Sources**:
    - When using `load_dataset` with `streaming=True`, the dataset is not downloaded upfront. Data is streamed on-the-fly. The `split_dataset_by_node` function works by having each rank iterate through the whole dataset and only yield the samples for its rank (using `itertools.islice`). This means every rank still iterates over the metadata for the full dataset, which can be slow for very large datasets.
    - True streaming from a source without a concept of shards (e.g., a web socket) would require custom logic to ensure that data is distributed among ranks and that checkpointing/resuming is possible. The Hugging Face `IterableDataset` from streaming has some support for this but needs careful handling.

4.  **Checkpointing and Resuming**:
    - Stateful iteration is crucial. The state of the dataset (which shard is being read, the position within the shard, the state of the shuffling RNG) must be saved in the checkpoint. The RFC correctly identifies this and proposes using `ds.state_dict()`.
    - When resuming, it's not enough to just load the model weights. The dataloader, including the dataset's internal state, must be restored. This ensures that training continues from the exact same data point, and data is not repeated or skipped.
    - Testing this is critical: a test should save a checkpoint mid-epoch, then start a new training run from that checkpoint and verify that the first batch of data is the one immediately following the last batch before checkpointing.

5.  **Shuffling**:
    - For `IterableDataset`, shuffling is often done with a shuffle buffer. The RFC mentions `ds.shuffle(shuffle_buffer_size, seed)`.
    - In a distributed setting, it's important that each rank shuffles its part of the data differently to ensure good training randomness. However, for reproducibility, the overall data stream should be the same given a seed.
    - The `set_epoch()` call is critical. It typically changes the seed for shuffling on each epoch, so the data order is different from epoch to epoch. Without it, the dataloader would yield the same sequence of data every epoch.

6.  **Divisibility Requirements**:
    - Be mindful of any implicit assumptions about divisibility. For example, some logic might assume that `num_shards` is perfectly divisible by `world_size`. The proposed `num_shards = world_size * num_shards_per_worker` elegantly handles this. However, if users provide their own sharded dataset, this could become an issue.

### Backward Compatibility

Options:

1. Make setup_data an utility, and have two utilities supporting old and new config formats.
   After deprecation period, old utility is removed.

   Pros:
   - Use it across recipes. Updates need to be done in one place.
   - Step towards our modularization goal.

   Cons:
   - Big change in how we handle recipe utilities

2. Create an adapter migrate_old_to_new_config:

   Pros:
   - Recipes still have method _setup_data exposing the logic

   Cons:
   - Hard to debug the migrated configs
   - Edge cases not covered by the adapter
   - ConcatDataset is handled differently

3. No migration. Old config with old recipe will break:
   - Users need to update their configs
   - Unknown impact on llamastack / startups / others

#### Implementation of Option 1 (Make setup_data an utility)

Location: torchtune/training/data_utils.py or similar

```python
@deprecated
def is_legacy_data_config(cfg: DictConfig) -> bool:
    """Detect if config follows legacy format vs new iterable dataset format."""
    # Check for new format indicators first
    has_dataloader_section = "dataloader" in cfg
    has_dataset_defaults = "dataset_defaults" in cfg
    has_dataset_setup = "dataset_setup" in cfg

    return not (has_dataloader_section or has_dataset_defaults or has_dataset_setup)

@deprecated
def setup_data_legacy(
    ...
) -> StatefulDataLoader:
    """
    Legacy data setup function to maintain backward compatibility.
    This replicates the current behavior in full_finetune_distributed.py
    """
    # same as current setup_data in the recipe....
    return dataloader
```

In the recipe:
```python
def _setup(...):
    ...
    if is_legacy_data_config(cfg):
        dataloader = setup_data_legacy(...)
    else:
        dataloader = setup_data(...)
```
