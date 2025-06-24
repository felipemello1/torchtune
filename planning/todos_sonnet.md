# Plan to Fix debug_iterable_dataset.py Issues

## MetricsAggregator Improvements

### 1. Documentation & Comments
**Issue**: Unclear documentation, state shape not explained
**Solution**: Add focused docstrings explaining:
- Why state uses tuple keys `(dataset_name, metric_name)` - enables multi-dataset namespacing
- Each aggregation type's internal state structure
- Distributed vs local computation paths

### 2. Replace has_value with None check
**Issue**: `state["has_value"]` is overkill for initialization tracking
**Solution**: 
- Initialize MAX/MIN with `state["value"] = None`
- Check `if state["value"] is not None` instead of separate boolean
- Simpler and more pythonic

### 3. Add sorting explanation
**Issue**: Unclear why we sort for distribution metrics
**Solution**: Add comment explaining sorting enables O(1) percentile lookup after O(n log n) sort
**Note**: This is standard percentile calculation - sort then index at percentage positions

### 4. Merge local metrics computation
**Issue**: `_compute_local_metrics` vs `_get_local_components` duplication
**Two approaches**:
a) Single method with `distributed=False` parameter (simpler)
b) Keep separate but extract common logic to `_process_metric_state()`

**Chosen**: Approach A - single method with parameter is cleaner and easier to follow

```python
def _compute_metrics(self, distributed: bool = False) -> Dict[str, Dict[str, Any]]:
    """Compute metrics with optional distributed reduction."""
    if not distributed:
        # Local computation path
        report = collections.defaultdict(dict)
        for (ds_name, metric_name), state in self._state.items():
            # ... existing local computation logic ...
        return dict(report)
    else:
        # Distributed path - prepare components for reduction
        components = self._prepare_reduction_components()
        return self._reduce_and_format_distributed(components)
```

### 5. Clarify "Build global component list"
**Issue**: Hard to understand component gathering
**Solution**: Add comment with example:
```python
# Example: [(('ds1', 'count'), 'value', 'sum', 5), (('ds1', 'max_val'), 'value', 'max', 10)]
# Format: (metric_key, field_name, reduction_op, local_value)
```

### 6. Empty unique_keys handling
**Issue**: When does `if not unique_keys:` happen?
**Solution**: This handles edge case where no rank contributed any metrics. Add comment explaining this prevents tensor creation errors.

### 7. Simplify device logic
**Issue**: Complex device selection logic
**Solution**: Default to CPU since metrics start there. Remove NCCL-specific device logic for now - can optimize later if needed.

### 8. Break down _reduce_and_format_distributed
**Issue**: Function too long and complex
**Solution**: Extract 3 functions:
- `_prepare_reduction_tensors()` - tensor creation and filling
- `_perform_distributed_reductions()` - actual reductions
- `_format_reduced_results()` - final formatting

```python
def _prepare_reduction_tensors(self, components: List[Tuple]) -> Dict[str, torch.Tensor]:
    """Prepare tensors for distributed reduction by operation type."""
    # Group components by reduction operation
    ops_data = {"sum": [], "max": [], "min": [], "mean": []}
    key_maps = {"sum": {}, "max": {}, "min": {}, "mean": {}}
    
    for metric_key, field_name, op_type, value in components:
        if op_type in ops_data:
            full_key = (metric_key, field_name)
            if full_key not in key_maps[op_type]:
                key_maps[op_type][full_key] = len(ops_data[op_type])
                ops_data[op_type].append(0.0)
            
            idx = key_maps[op_type][full_key]
            ops_data[op_type][idx] = value
    
    # Convert to tensors
    tensors = {}
    for op_type, values in ops_data.items():
        if values:
            tensors[op_type] = torch.tensor(values, device="cpu")
    
    return tensors, key_maps

def _perform_distributed_reductions(self, tensors: Dict[str, torch.Tensor]) -> None:
    """Execute async distributed reductions."""
    handles = []
    
    if "sum" in tensors:
        handles.append(dist.all_reduce(tensors["sum"], op=dist.ReduceOp.SUM, async_op=True))
    if "max" in tensors:
        handles.append(dist.all_reduce(tensors["max"], op=dist.ReduceOp.MAX, async_op=True))
    if "min" in tensors:
        handles.append(dist.all_reduce(tensors["min"], op=dist.ReduceOp.MIN, async_op=True))
    if "mean" in tensors:
        handles.append(dist.all_reduce(tensors["mean"], op=dist.ReduceOp.SUM, async_op=True))
    
    # Wait for all reductions
    for handle in handles:
        handle.wait()
    
    # Apply mean division
    if "mean" in tensors:
        tensors["mean"] /= self._world_size

def _format_reduced_results(self, tensors: Dict, key_maps: Dict) -> Dict[str, Dict[str, Any]]:
    """Format reduced tensor results into final metrics dictionary."""
    report = collections.defaultdict(dict)
    
    for op_type, tensor in tensors.items():
        for (metric_key, field_name), idx in key_maps[op_type].items():
            ds_name, metric_name = metric_key
            state = self._state[metric_key]
            agg_type = state["type"]
            
            if agg_type == AggregationType.SUM:
                report[ds_name][metric_name] = tensor[idx].item()
            elif agg_type == AggregationType.MEAN and field_name == "sum":
                # Handle mean calculation
                count_key = (metric_key, "count")
                count_idx = key_maps["sum"].get(count_key)
                if count_idx is not None:
                    count_val = tensors["sum"][count_idx].item()
                    if count_val > 0:
                        report[ds_name][metric_name] = tensor[idx].item() / count_val
            # ... other aggregation types ...
    
    return dict(report)
```

### 9. Cache world_size
**Issue**: Multiple `get_world_size()` calls
**Solution**: Store as instance variable `self._world_size` set once in `get_metrics()`

### 10. Consistent metric naming
**Issue**: Inconsistent p05/p50 vs metric names
**Solution**: Use original metric names as keys: `report[ds_name][metric_name] = data["value"]`

### 11. Simplify state_dict serialization
**Issue**: Complex loops and conditionals for serialization
**Two approaches**:
a) Custom serialization only for non-serializable types (deque, Counter)
b) Convert all to basic types upfront

**Chosen**: Approach A - only handle deque→list and Counter→dict, keep rest as-is

```python
def state_dict(self) -> Dict[str, Any]:
    """Serialize aggregator state for checkpointing."""
    serializable_state = {}
    
    for key, state in self._state.items():
        serializable_state[key] = state.copy()
        
        # Handle non-serializable types
        if "values" in state:  # deque for distribution metrics
            serializable_state[key]["values"] = list(state["values"])
        if "counts" in state:  # Counter for categorical metrics
            serializable_state[key]["counts"] = dict(state["counts"])
    
    return {
        "state": serializable_state,
        "dist_window_size": self._dist_window_size
    }

def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
    """Load aggregator state from checkpoint."""
    self._dist_window_size = state_dict["dist_window_size"]
    self._state = {}
    
    for key, state in state_dict["state"].items():
        # Convert string keys back to tuples if needed
        key = ast.literal_eval(key) if isinstance(key, str) else key
        self._state[key] = state.copy()
        
        # Restore non-serializable types
        if "values" in state:
            self._state[key]["values"] = collections.deque(
                state["values"], maxlen=self._dist_window_size
            )
        if "counts" in state:
            self._state[key]["counts"] = collections.Counter(state["counts"])
```

## HFIterableDataset Improvements

### 12. Document transform pipeline
**Issue**: Unclear why metric_transform separate from others
**Solution**: Add comment explaining HF .map() incompatibility forces separate application in `__iter__`

### 13. Remove error handling (DONE)
**Issue**: Unnecessary complexity with _filter_failed_transforms
**Action**: Verify no tests depend on this, remove all related code

### 14. Simplify state_dict responsibility  
**Issue**: Datasets shouldn't wrap state with their names
**Solution**: Change interface - datasets return plain state dict, parent handles namespacing
**Impact**: Update InterleavedDataset to handle child state namespacing

```python
# HFIterableDataset - OLD approach
def state_dict(self) -> Dict[str, Any]:
    state = {
        "num_epochs": self._num_epochs,
        "seed": self._seed,
        "hf_dataset_state": self._ds.state_dict(),
    }
    return {self.dataset_name: state}  # ❌ Dataset wraps with name

# HFIterableDataset - NEW approach  
def state_dict(self) -> Dict[str, Any]:
    return {
        "num_epochs": self._num_epochs,
        "seed": self._seed,
        "hf_dataset_state": self._ds.state_dict(),
    }  # ✅ Just return state

def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
    # Expect plain state dict, not wrapped
    self._num_epochs = state_dict["num_epochs"]
    self._ds.load_state_dict(state_dict["hf_dataset_state"])

# InterleavedDataset - handle namespacing for children
def state_dict(self) -> Dict[str, Any]:
    child_states = {}
    for ds in self._datasets:
        child_states[ds.dataset_name] = ds.state_dict()  # ✅ Parent handles namespacing
    
    return {
        "seed": self._seed,
        "sampling_generator_state": self._sampling_generator.get_state(),
        "child_states": child_states,
    }

def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
    self._sampling_generator.set_state(state_dict["sampling_generator_state"])
    
    child_states = state_dict["child_states"]
    for ds in self._datasets:
        if ds.dataset_name in child_states:
            ds.load_state_dict(child_states[ds.dataset_name])  # ✅ Pass plain state
```

### 15. Remove unused _seed in InterleavedDataset
**Issue**: _seed only used in __init__ 
**Solution**: Remove instance variable, use local variable in __init__

### 16. Remove dead variables
**Identified for deletion**:
- `_max_transform_failures_per_epoch` and `_transform_failures_this_epoch` (unused after error handling removal)
- Any logging variables only used for debugging

### 17. Make InterleavedDataset safer
**Issue**: Array indexing fragile
**Solution**: Use dictionary `{ds.dataset_name: ds}` instead of list, safer lookup

```python
# OLD approach - fragile indexing
class InterleavedDataset:
    def __init__(self, datasets: List[TuneIterableDataset], weights: List[float], ...):
        self._datasets = datasets  # ❌ List indexing
        self._weights = torch.tensor(weights)
    
    def __iter__(self):
        child_iters = [iter(ds) for ds in self._datasets]  # ❌ Parallel arrays
        while True:
            ds_idx = torch.multinomial(self._weights, 1).item()
            sample = next(child_iters[ds_idx])  # ❌ Index into parallel structure

# NEW approach - dictionary-based
class InterleavedDataset:
    def __init__(self, datasets: List[TuneIterableDataset], weights: List[float], ...):
        # Create mapping and preserve order
        self._datasets = {ds.dataset_name: ds for ds in datasets}  # ✅ Dictionary lookup
        self._dataset_names = [ds.dataset_name for ds in datasets]  # ✅ Ordering preserved
        self._weights = torch.tensor(weights)
    
    def __iter__(self):
        child_iters = {name: iter(ds) for name, ds in self._datasets.items()}  # ✅ Safe lookup
        
        while True:
            idx = torch.multinomial(self._weights, 1).item()
            selected_name = self._dataset_names[idx]  # ✅ Name-based selection
            sample = next(child_iters[selected_name])  # ✅ Dictionary lookup
            yield sample
```

## Test Infrastructure Improvements

### 18. Simplify collate function
**Issue**: Too complex for test purposes
**Solution**: Basic collate that just extracts metrics and pads tokens - remove all special handling

### 19. Improve run_training_loop
**Issue**: Takes iterator instead of dataloader, requires two calls for checkpointing
**Solution**: 
```python
def run_training_loop(
    dataloader, 
    aggregator: MetricsAggregator, 
    num_steps: int, 
    checkpoint_at: Optional[int] = None
) -> Tuple[List[Dict], Dict[str, Dict], Optional[Dict]]:
    """
    Run training loop with optional checkpointing.
    
    Returns:
        (collected_batches, final_metrics, checkpoint_state_or_none)
    """
    collected_batches = []
    checkpoint_state = None
    
    for i, batch in enumerate(islice(dataloader, num_steps)):
        # Save checkpoint at specified step
        if checkpoint_at is not None and i == checkpoint_at:
            checkpoint_state = {
                'loader_state': dataloader.state_dict(),
                'aggregator_state': aggregator.state_dict()
            }
        
        # Process batch
        if "metrics" in batch:
            aggregator.update(batch.pop("metrics"))
        collected_batches.append(batch)
    
    final_metrics = aggregator.get_metrics()
    return collected_batches, final_metrics, checkpoint_state

# Usage in tests becomes much cleaner:
# OLD way - two separate calls
batches1, metrics1 = run_training_loop(iter(loader), aggregator, 10)
checkpoint_state = {'loader': loader.state_dict(), 'agg': aggregator.state_dict()}
batches2, metrics2 = run_training_loop(iter(loader), aggregator, 5)

# NEW way - single call with checkpoint
batches1, metrics1, checkpoint_state = run_training_loop(loader, aggregator, 15, checkpoint_at=10)
```

### 20. Remove assert_sample_structure
**Issue**: Used only once, not core functionality
**Solution**: Delete utility, inline check if needed

### 21. Delete assert_checkpoint_continuation
**Issue**: Complex utility not adding value
**Solution**: Remove entirely, use simple direct assertions

### 22. Fix StandardMetricTransform dataset_name
**Issue**: Requiring dataset_name in constructor is inconvenient
**Two approaches**:
a) Add `set_dataset_name()` method called by dataset
b) Make dataset_name a property that gets set during transform application

**Chosen**: Approach A - explicit method call is clearer

```python
# OLD approach - constructor injection
class StandardMetricTransform:
    def __init__(self, dataset_name: str):  # ❌ Required in constructor
        self.dataset_name = dataset_name
        self.new_metric = partial(Metric, dataset_name=dataset_name)

# Usage - inconvenient for configs
transform = StandardMetricTransform("my_dataset")  # ❌ Hard to configure

# NEW approach - explicit method
class StandardMetricTransform:
    """
    Standard metrics transform for training datasets.
    
    Tracks:
    - samples_seen: Count of samples processed (SUM aggregation)
    - tokens_seen: Total tokens across samples (SUM aggregation) 
    - seq_len: Sequence length distribution (DISTRIBUTION aggregation)
    
    Designed for multi-worker DataLoader and distributed training environments.
    """
    
    def __init__(self):
        self.dataset_name: Optional[str] = None
        self.new_metric: Optional[Callable] = None
    
    def set_dataset_name(self, dataset_name: str) -> None:
        """Set dataset name for metric namespacing. Called by dataset."""
        self.dataset_name = dataset_name
        self.new_metric = partial(Metric, dataset_name=dataset_name)
    
    def __call__(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        if self.dataset_name is None:
            raise RuntimeError("set_dataset_name() must be called before using transform")
        
        # Extract metrics as before...
        return sample

# Usage in dataset
class HFIterableDataset:
    def __init__(self, metric_transform=None, **kwargs):
        self._metric_transform = metric_transform or StandardMetricTransform()
        # Set dataset name after we know it
        if hasattr(self._metric_transform, 'set_dataset_name'):
            self._metric_transform.set_dataset_name(self.dataset_name)  # ✅ Clean injection
```

### 23. Document StandardMetricTransform
**Solution**: Add docstring explaining:
- Metrics tracked: samples_seen (count), tokens_seen (sum), seq_len (distribution)
- Designed for multi-worker, multi-rank aggregation
- Why these specific metrics matter for training

### 24. Factory pattern alternatives
**Considered**: Builder pattern, direct instantiation
**Decision**: Keep factory - provides good defaults and test isolation

## Test Simplification

### 25. Simplify test_epoch_boundaries
**Solution**: Single dataset (small), test exact epoch completion:
- 1 epoch: collect all samples, verify count equals dataset size
- 2 epochs: verify first N samples repeat after dataset_size
- 3 epochs: same pattern

```python
@pytest.mark.parametrize("num_epochs", [1, 2, 3])
def test_epoch_boundaries(self, num_epochs, dataset_factory, small_dataset_file):
    """Test that dataset correctly completes exact number of epochs."""
    dataset = dataset_factory(small_dataset_file, shuffle=False)
    
    total_samples = SMALL_DATASET_SIZE * num_epochs
    samples = list(islice(iter(dataset), total_samples))
    
    # Verify we got expected number of samples
    assert len(samples) == total_samples
    
    # Verify epoch repetition - every SMALL_DATASET_SIZE samples should repeat
    for epoch in range(1, num_epochs):
        epoch_start = epoch * SMALL_DATASET_SIZE
        first_epoch_ids = [s["id"] for s in samples[:SMALL_DATASET_SIZE]]
        current_epoch_ids = [s["id"] for s in samples[epoch_start:epoch_start + SMALL_DATASET_SIZE]]
        
        assert first_epoch_ids == current_epoch_ids, \
            f"Epoch {epoch} data should repeat epoch 0 data"
    
    # Verify samples are sequential within each epoch (no shuffle)
    for epoch in range(num_epochs):
        epoch_start = epoch * SMALL_DATASET_SIZE
        epoch_samples = samples[epoch_start:epoch_start + SMALL_DATASET_SIZE]
        epoch_ids = [s["id"] for s in epoch_samples]
        expected_ids = list(range(SMALL_DATASET_SIZE))  # 0, 1, 2, ..., SMALL_DATASET_SIZE-1
        
        assert epoch_ids == expected_ids, \
            f"Epoch {epoch} should have sequential IDs without shuffle"
```

### 26. Simplify test_shuffling_behavior
**Solution**: Two datasets, one shuffled, one not:
- Non-shuffled: epoch 1 == epoch 2 sample order
- Shuffled: epoch 1 != epoch 2 sample order, but same sample set

```python
def test_shuffling_behavior(self, dataset_factory, small_dataset_file):
    """Test shuffling changes order between epochs but preserves sample set."""
    # Non-shuffled dataset
    dataset_no_shuffle = dataset_factory(small_dataset_file, shuffle=False)
    no_shuffle_samples = list(islice(iter(dataset_no_shuffle), SMALL_DATASET_SIZE * 2))
    
    epoch1_ids = [s["id"] for s in no_shuffle_samples[:SMALL_DATASET_SIZE]]
    epoch2_ids = [s["id"] for s in no_shuffle_samples[SMALL_DATASET_SIZE:]]
    
    # Non-shuffled: epochs should be identical
    assert epoch1_ids == epoch2_ids, "Non-shuffled epochs should have identical order"
    assert epoch1_ids == list(range(SMALL_DATASET_SIZE)), "Should be sequential"
    
    # Shuffled dataset
    dataset_shuffle = dataset_factory(small_dataset_file, shuffle=True)
    shuffle_samples = list(islice(iter(dataset_shuffle), SMALL_DATASET_SIZE * 2))
    
    shuffle_epoch1_ids = [s["id"] for s in shuffle_samples[:SMALL_DATASET_SIZE]]
    shuffle_epoch2_ids = [s["id"] for s in shuffle_samples[SMALL_DATASET_SIZE:]]
    
    # Shuffled: epochs should have different order
    assert shuffle_epoch1_ids != shuffle_epoch2_ids, "Shuffled epochs should have different order"
    
    # But same sample sets
    assert set(shuffle_epoch1_ids) == set(range(SMALL_DATASET_SIZE)), "Epoch 1 should have all samples"
    assert set(shuffle_epoch2_ids) == set(range(SMALL_DATASET_SIZE)), "Epoch 2 should have all samples"
    
    # Shuffled should differ from non-shuffled (with high probability)
    assert shuffle_epoch1_ids != epoch1_ids, "Shuffled should differ from sequential"
```

### 27. Delete test_transform_error_handling (DONE)

### 28. Use float epochs in test_checkpointing
**Solution**: `@pytest.mark.parametrize("num_epochs", [0.5, 1.0, 2.5])`
Calculate samples = `int(dataset_size * num_epochs)`

### 29. Verify weight normalization
**Solution**: In test_initialization_validation, assert `sum(interleaved._weights) == 1.0`

### 30. Default to small_dataset_size
**Solution**: Use SMALL_DATASET_SIZE unless test specifically needs larger

### 31. Test dataset_name auto-generation
**Solution**: Test that path="json" + split="train" → dataset_name="json_train"

### 32. Test epoch mismatch in metrics
**Solution**: Interleave datasets with different speeds, verify epoch counters differ

### 33. Simplify test_sampling_ratios
**Solution**:
```python
def test_sampling_ratios(self, dataset_factory, small_dataset_file, medium_dataset_file):
    """Test that interleaved dataset samples according to specified weights."""
    # ds1: IDs 0-22 (small), ds2: IDs 100-134 (medium, offset=100)
    ds1 = dataset_factory(small_dataset_file, dataset_name="ds1")
    ds2 = dataset_factory(medium_dataset_file, dataset_name="ds2") 
    
    weights = [0.3, 0.7]  # 30% ds1, 70% ds2
    interleaved = InterleavedDataset([ds1, ds2], weights, seed=SEED)
    
    # Collect samples and categorize by ID range
    sample_count = 500
    samples = list(islice(iter(interleaved), sample_count))
    
    ds1_count = sum(1 for s in samples if s["id"] < 50)    # IDs 0-22 from ds1
    ds2_count = sum(1 for s in samples if s["id"] >= 100)  # IDs 100+ from ds2
    total_categorized = ds1_count + ds2_count
    
    # Verify we categorized all samples correctly
    assert total_categorized == sample_count, "All samples should be categorized"
    
    # Check ratios with tolerance
    ds1_ratio = ds1_count / sample_count
    ds2_ratio = ds2_count / sample_count
    
    tolerance = 0.05  # 5% tolerance
    assert abs(ds1_ratio - weights[0]) < tolerance, \
        f"ds1 ratio {ds1_ratio:.3f} should be ~{weights[0]} (count: {ds1_count})"
    assert abs(ds2_ratio - weights[1]) < tolerance, \
        f"ds2 ratio {ds2_ratio:.3f} should be ~{weights[1]} (count: {ds2_count})"
```

### 34. TestEndToEndCheckpointing Decision
**Two options analyzed**:

**Option A: Delete and distribute tests**
Tests to add to existing classes:
- TestHFIterableDataset: distributed epoch boundaries test  
- TestInterleavedDataset: distributed sampling test
- TestDistributedMetricsAggregator: checkpoint restore test

**Option B: Simplify significantly**
```python
@pytest.mark.parametrize("num_epochs", [0.5, 1.0, 2.5])
@pytest.mark.parametrize("num_workers", [0, 3])
def test_interleaved_checkpointing(self, num_epochs, num_workers, dataset_factory, tmp_data_dir):
    """Test checkpointing with interleaved dataset across different configurations."""
    # Create datasets
    file1 = tmp_data_dir / "ds1.json" 
    file2 = tmp_data_dir / "ds2.json"
    create_test_json_file(file1, SMALL_DATASET_SIZE)
    create_test_json_file(file2, SMALL_DATASET_SIZE, offset=100)
    
    ds1 = dataset_factory(str(file1), dataset_name="ds1", shuffle=False)
    ds2 = dataset_factory(str(file2), dataset_name="ds2", shuffle=False)
    interleaved = InterleavedDataset([ds1, ds2], [0.5, 0.5], seed=SEED)
    
    # Create dataloader
    loader = StatefulDataLoader(
        interleaved,
        batch_size=BATCH_SIZE,
        num_workers=num_workers,
        collate_fn=collate_with_metrics,
    )
    aggregator = MetricsAggregator()
    
    # Calculate steps for given epochs
    samples_per_epoch = SMALL_DATASET_SIZE * 2  # Two datasets
    total_samples = int(samples_per_epoch * num_epochs)
    steps_to_checkpoint = max(1, total_samples // (BATCH_SIZE * 2))  # Checkpoint at halfway
    total_steps = max(2, total_samples // BATCH_SIZE)
    
    # Run with checkpointing
    batches1, metrics1, checkpoint_state = run_training_loop(
        loader, aggregator, total_steps, checkpoint_at=steps_to_checkpoint
    )
    
    # Continue original run  
    continuation_batches, final_metrics = run_training_loop(loader, aggregator, 3)
    
    # Restore from checkpoint
    loader2 = StatefulDataLoader(interleaved, batch_size=BATCH_SIZE, num_workers=num_workers, collate_fn=collate_with_metrics)
    aggregator2 = MetricsAggregator()
    
    loader2.load_state_dict(checkpoint_state['loader_state'])
    aggregator2.load_state_dict(checkpoint_state['aggregator_state'])
    
    # Resume from checkpoint
    resumed_batches, resumed_metrics = run_training_loop(loader2, aggregator2, 3)
    
    # Verification
    assert len(continuation_batches) == len(resumed_batches), "Should process same number of batches"
    
    # For single worker + no shuffle, exact match expected
    if num_workers == 0:
        cont_ids = [b["id"].tolist() for b in continuation_batches]
        resumed_ids = [b["id"].tolist() for b in resumed_batches]
        assert cont_ids == resumed_ids, "Single worker should have exact continuation"
    
    # Metrics should always match (distributed reduction handles aggregation)
    assert final_metrics == resumed_metrics, "Metrics should match after restoration"
```

**Chosen**: Option B + delete TestDistributedDataLoading (highly duplicative)

### 35. Remove unnecessary tmp_dir complexity
**Issue**: Unclear why rank-specific directories needed
**Solution**: Use single temp directory unless we identify actual race condition

### 36. Complete test_distributed_aggregation
**Issue**: Missing metric types compared to single-device tests
**Solution**: Test all AggregationType values, not just SUM/MAX

### 37. Remove duplicate dataset_factory methods
**Issue**: Each test class has own factory
**Solution**: Use single fixture, remove class methods

### 38. Delete TestDistributedEdgeCases
**Issue**: Adds complexity without clear value
**Solution**: Remove entirely

## Implementation Priority
1. MetricsAggregator core fixes (1-11)
2. Dataset interface changes (12-17) 
3. Test infrastructure (18-24)
4. Test simplification (25-38)

## Open Questions
1. Should we keep the deque maxlen for distribution metrics, or use unlimited list?
2. For distributed percentiles, is averaging rank percentiles mathematically correct vs global sort?
3. Do we need explicit testing for very large datasets that might hit memory limits?

## Concerns & Blind Spots
1. Removing error handling might cause silent failures - should add basic logging
2. State dict interface change requires careful migration of any existing checkpoints
3. Some distributed tests might need NCCL backend specifics we're removing
4. Factory pattern removal might break user code that depends on current interface

## Notes
- All changes maintain backward compatibility where possible
- Focus on readability and maintainability over micro-optimizations
- Tests should be simple enough that their intent is obvious
