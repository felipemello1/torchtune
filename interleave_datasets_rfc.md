# RFC: Framework-Agnostic Interleave Datasets

This document proposes the design and implementation of a custom, framework-agnostic `interleave_datasets` utility. This utility will merge multiple iterable-style datasets into a single stream, with options for weighted sampling and different stopping behaviors.

## 1. Motivation

The current reliance on `datasets.interleave_datasets` creates a dependency on the Hugging Face `datasets` library, limiting portability and flexibility. A custom implementation offers several advantages:

*   **Framework Agnostic**: Works with any Python iterable, not just `datasets.IterableDataset`. This allows for broader use with different data loading pipelines (e.g., PyTorch's `IterableDataset`, custom data generators).
*   **Simplified Logic**: Provides a clear and self-contained implementation without the overhead and complexity of the larger `datasets` library, making it easier to debug and maintain.
*   **No External Dependencies**: Reduces the project's dependency footprint.

The goal is to create a utility that is functionally equivalent for our needs but is decoupled from any specific data-loading library.

## 2. API Design

The function will have the following signature:

```python
from typing import Iterable, List, Optional

def interleave_datasets(
    datasets: List[Iterable],
    probabilities: List[float],
    seed: Optional[int] = None,
    stopping_strategy: str = "first_exhausted",
) -> Iterable:
    """
    Interleaves multiple iterable datasets into a single stream with weighted sampling.

    Args:
        datasets (List[Iterable]): A list of iterable datasets to interleave.
        probabilities (List[float]): A list of probabilities corresponding to each
            dataset, used for weighted sampling. Must sum to 1.0.
        seed (Optional[int]): Seed for the random number generator to ensure
            reproducibility.
        stopping_strategy (str): Determines when to stop iteration.
            - "first_exhausted": Stops as soon as any single dataset is exhausted.
            - "all_exhausted": Continues until all datasets are exhausted.

    Yields:
        An item from one of the datasets.
    """
```

## 3. Implementation Details

The core of the function will be a generator that maintains iterators for each of the input datasets. On each step, it will use a weighted random choice to select which dataset to sample from next.

### Core Logic

1.  **Initialization**:
    *   Validate that the `datasets` and `probabilities` lists are of the same length and that probabilities sum to 1.0.
    *   Create a list of iterators from the input `datasets`.
    *   Initialize a `random.Random` instance with the given `seed` for reproducibility.

2.  **Sampling Loop**:
    *   The main loop will continue as long as there are active (non-exhausted) datasets.
    *   Inside the loop, it will use `random.choices` (or a custom implementation) to select a dataset index based on the provided `probabilities`.
    *   It will then attempt to get the next item from the chosen dataset's iterator using `next()`.

3.  **Handling Exhausted Datasets**:
    *   If a `StopIteration` exception is caught, the corresponding dataset is considered exhausted.
    *   The exhausted dataset, its iterator, and its probability are removed from the active lists.
    *   The remaining probabilities are **renormalized** to ensure they still sum to 1.0, maintaining the relative weights of the remaining active datasets.

### Stopping Strategies

*   **`first_exhausted`**: The `while` loop's condition will simply be `while datasets:`. As soon as the first `StopIteration` is caught and a dataset is removed, the loop for the next iteration will check the (now smaller) list. If the `stopping_strategy` is set to `first_exhausted`, the loop will terminate immediately after the first dataset is exhausted. In practice, this means the generator will be exhausted and will stop yielding.

*   **`all_exhausted`**: The `while datasets:` condition naturally handles this. The loop continues as long as there is at least one active dataset. When a dataset is exhausted, it's removed, and the loop continues with the remaining ones until no datasets are left.

## 4. Example Implementation

Here is a potential implementation:

```python
import random
from typing import Iterable, List, Optional

def interleave_datasets(
    datasets: List[Iterable],
    probabilities: List[float],
    seed: Optional[int] = None,
    stopping_strategy: str = "first_exhausted",
) -> Iterable:
    """
    Interleaves multiple iterable datasets into a single stream with weighted sampling.
    """
    if not datasets:
        return

    if len(datasets) != len(probabilities):
        raise ValueError("The number of datasets and probabilities must be the same.")

    if not abs(sum(probabilities) - 1.0) < 1e-6:
        raise ValueError("Probabilities must sum to 1.0.")

    if stopping_strategy not in ["first_exhausted", "all_exhausted"]:
        raise ValueError("stopping_strategy must be 'first_exhausted' or 'all_exhausted'.")

    rng = random.Random(seed)
    iterators = [iter(ds) for ds in datasets]

    # Keep track of active iterators and their corresponding probabilities
    active_iterators = list(range(len(iterators)))
    active_probabilities = list(probabilities)

    while active_iterators:
        # Choose an iterator index based on current weights
        chosen_idx_of_idx = rng.choices(
            range(len(active_iterators)), weights=active_probabilities, k=1
        )[0]
        original_idx = active_iterators[chosen_idx_of_idx]
        iterator = iterators[original_idx]

        try:
            yield next(iterator)
        except StopIteration:
            if stopping_strategy == "first_exhausted":
                return  # Stop as soon as one is exhausted

            # Remove the exhausted iterator and its probability
            active_iterators.pop(chosen_idx_of_idx)
            active_probabilities.pop(chosen_idx_of_idx)

            # Renormalize the probabilities of the remaining active iterators
            if active_probabilities:
                total_prob = sum(active_probabilities)
                if total_prob > 0:
                    active_probabilities = [p / total_prob for p in active_probabilities]
                else:
                    # If remaining probabilities are all zero, distribute equally
                    active_probabilities = [1.0 / len(active_probabilities)] * len(active_probabilities)

# --- Example Usage ---

def create_generator(name, count):
    """A simple generator for demonstration."""
    for i in range(count):
        yield f"{name}-{i}"

# Datasets with different lengths
dataset1 = create_generator("A", 5)  # 5 samples
dataset2 = create_generator("B", 10) # 10 samples
dataset3 = create_generator("C", 3)  # 3 samples

# Interleave with "first_exhausted"
print("--- Strategy: first_exhausted ---")
interleaved_stream = interleave_datasets(
    datasets=[dataset1, dataset2, dataset3],
    probabilities=[0.6, 0.3, 0.1],
    seed=42,
    stopping_strategy="first_exhausted",
)
# This will stop after C is likely exhausted, or one of the others.
# The total number of yielded items will be small.
for item in interleaved_stream:
    print(item)

# Interleave with "all_exhausted"
print("\n--- Strategy: all_exhausted ---")
# Re-create generators since they are exhausted
dataset1 = create_generator("A", 5)
dataset2 = create_generator("B", 10)
dataset3 = create_generator("C", 3)
interleaved_stream_all = interleave_datasets(
    datasets=[dataset1, dataset2, dataset3],
    probabilities=[0.6, 0.3, 0.1],
    seed=42,
    stopping_strategy="all_exhausted",
)
# This will continue until all 18 (5 + 10 + 3) items are yielded.
count = 0
for item in interleaved_stream_all:
    print(item, end=' ')
    count += 1
print(f"\nTotal items: {count}")

```

This design provides a robust, reusable, and framework-agnostic solution for interleaving datasets, directly addressing the requirements while promoting modularity and reducing dependencies. 