"""
Bug reproduction: StatefulDataLoader + HuggingFace IterableDataset skips samples on resume

This demonstrates the issue when using PyTorch's StatefulDataLoader with 
HuggingFace IterableDataset that has transformations applied.
"""

from datasets import Dataset
from torch.utils.data import IterableDataset as TorchIterableDataset
from torchdata.stateful_dataloader import StatefulDataLoader
import torch

# Wrapper to make HF dataset compatible with PyTorch
class HFDatasetWrapper(TorchIterableDataset):
    def __init__(self, hf_dataset):
        self.hf_dataset = hf_dataset
    
    def __iter__(self):
        return iter(self.hf_dataset)
    
    def state_dict(self):
        return self.hf_dataset.state_dict()
    
    def load_state_dict(self, state_dict):
        self.hf_dataset.load_state_dict(state_dict)

# Create HuggingFace dataset with transformation
hf_ds = Dataset.from_dict({"id": list(range(50))})
hf_ds = hf_ds.to_iterable_dataset(num_shards=4)
hf_ds = hf_ds.map(lambda x: {"id": x["id"], "tokens": [x["id"]] * 3})  # Transformation

# Wrap for PyTorch
dataset = HFDatasetWrapper(hf_ds)

# Create StatefulDataLoader
dataloader = StatefulDataLoader(dataset, batch_size=5, num_workers=0)

# Process some batches and checkpoint
batches_before_checkpoint = []
for i, batch in enumerate(dataloader):
    batches_before_checkpoint.append(batch["id"].tolist())
    if i == 1:  # Checkpoint after 2 batches (10 samples)
        checkpoint = dataloader.state_dict()
        print(f"Checkpointed after batch {i}, last sample id: {batch['id'][-1].item()}")
        break

# Continue with original dataloader
original_continuation = []
for i, batch in enumerate(dataloader):
    original_continuation.extend(batch["id"].tolist())
    if i >= 0:  # Just get next batch
        break

# Create new dataloader and restore
hf_ds2 = Dataset.from_dict({"id": list(range(50))})
hf_ds2 = hf_ds2.to_iterable_dataset(num_shards=4)
hf_ds2 = hf_ds2.map(lambda x: {"id": x["id"], "tokens": [x["id"]] * 3})
dataset2 = HFDatasetWrapper(hf_ds2)
dataloader2 = StatefulDataLoader(dataset2, batch_size=5, num_workers=0)

# Load checkpoint
dataloader2.load_state_dict(checkpoint)

# Get continuation from restored dataloader
resumed_continuation = []
for i, batch in enumerate(dataloader2):
    resumed_continuation.extend(batch["id"].tolist())
    if i >= 0:  # Just get next batch
        break

print(f"\nProcessed before checkpoint: {batches_before_checkpoint}")
print(f"\nOriginal continuation: {original_continuation}")
print(f"Resumed continuation:  {resumed_continuation}")

if original_continuation != resumed_continuation:
    print(f"\n❌ BUG DETECTED: Samples were skipped!")
    print(f"   Expected to continue from sample {original_continuation[0]}")
    print(f"   But resumed from sample {resumed_continuation[0]}")
    print(f"   Skipped {resumed_continuation[0] - original_continuation[0]} samples")
else:
    print("\n✅ No bug detected") 