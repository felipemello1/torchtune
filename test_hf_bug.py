from datasets import Dataset
from datasets.distributed import split_dataset_by_node

# Create a larger dataset to ensure we hit batch boundaries
ds = Dataset.from_dict({"n": list(range(100))})
ds = ds.to_iterable_dataset(num_shards=4)

world_size = 2
rank = 0
ds_rank = split_dataset_by_node(ds, rank, world_size)

# First, collect some samples and checkpoint
it = iter(ds_rank)
examples_before_checkpoint = []
checkpoint_idx = 9  # Try different values like 2, 5, 9, etc.

for idx, example in enumerate(it):
    examples_before_checkpoint.append(example)
    if idx == checkpoint_idx:
        state_dict = ds_rank.state_dict()
        print(f"Checkpointing at index {idx}, example: {example}")
        break

# Continue with the ORIGINAL iterator
examples_original_continuation = []
for idx, example in enumerate(it):
    examples_original_continuation.append(example)
    if idx >= 5:  # Get a few more samples
        break

# Now create a new dataset and restore from checkpoint
ds2 = Dataset.from_dict({"n": list(range(100))})
ds2 = ds2.to_iterable_dataset(num_shards=4)
ds2_rank = split_dataset_by_node(ds2, rank, world_size)
ds2_rank.load_state_dict(state_dict)

# Get samples from restored iterator
it_resumed = iter(ds2_rank)
examples_resumed_continuation = []
for idx, example in enumerate(it_resumed):
    examples_resumed_continuation.append(example)
    if idx >= 5:  # Same number as original continuation
        break

print("\n=== RESULTS ===")
print(f"Examples before checkpoint: {examples_before_checkpoint}")
print(f"\nOriginal iterator continuation: {examples_original_continuation}")
print(f"Resumed iterator continuation: {examples_resumed_continuation}")

# Check if they match
if examples_original_continuation == examples_resumed_continuation:
    print("\n✓ Continuations match - no bug detected")
else:
    print("\n✗ Continuations DO NOT match - bug reproduced!")
    print(f"Original first: {examples_original_continuation[0] if examples_original_continuation else 'None'}")
    print(f"Resumed first: {examples_resumed_continuation[0] if examples_resumed_continuation else 'None'}")

# Also try with formatting/mapping which might trigger the issue
print("\n\n=== Testing with map transformation ===")

def format_func(example):
    return {"n": example["n"], "n_squared": example["n"] ** 2}

ds3 = Dataset.from_dict({"n": list(range(100))})
ds3 = ds3.to_iterable_dataset(num_shards=4)
ds3 = ds3.map(format_func)  # Add formatting
ds3_rank = split_dataset_by_node(ds3, rank, world_size)

it3 = iter(ds3_rank)
examples3_before = []
for idx, example in enumerate(it3):
    examples3_before.append(example)
    if idx == checkpoint_idx:
        state_dict3 = ds3_rank.state_dict()
        break

# Original continuation
examples3_original = []
for idx, example in enumerate(it3):
    examples3_original.append(example)
    if idx >= 5:
        break

# Resumed continuation
ds4 = Dataset.from_dict({"n": list(range(100))})
ds4 = ds4.to_iterable_dataset(num_shards=4)
ds4 = ds4.map(format_func)
ds4_rank = split_dataset_by_node(ds4, rank, world_size)
ds4_rank.load_state_dict(state_dict3)

it4_resumed = iter(ds4_rank)
examples4_resumed = []
for idx, example in enumerate(it4_resumed):
    examples4_resumed.append(example)
    if idx >= 5:
        break

print(f"\nWith formatting - Original continuation: {[e['n'] for e in examples3_original]}")
print(f"With formatting - Resumed continuation: {[e['n'] for e in examples4_resumed]}")

if examples3_original == examples4_resumed:
    print("✓ Formatted continuations match")
else:
    print("✗ Formatted continuations DO NOT match - bug reproduced!") 