### Plan to Refactor `debug_iterable_dataset.py`

This document provides a consolidated and detailed plan to refactor `debug_iterable_dataset.py`. It synthesizes the best proposals from three different LLMs (Gemini, Opus, Sonnet) and includes the specific code examples and structural changes required for implementation. It addresses all 38 points from the original request, focusing on creating simple, elegant, and robust solutions.

Tasks:
1. MetricsAggregator unclear documentation. Add docstrings and comments, but keep it short. Avoid stating obvious things. Use the space to add important unclear insights. Unclear why state has that shape.
2. MetricsAggregator: unclear why "state["has_value"]" exists. I know its there to avoid having to deal with max(-inf, num) if the number is not initialized. But adding another arg "has_value" is an overkill. Just check if value is not None
3. MetricsAggregator: Unclear why we sort. I know its so its easier to get the percentiles, max, min. But add the note. By the way, is this a fair way to compute percentiles?
4. MetricsAggregator: merge _compute_local_metrics/_get_local_components. The issue is that _get_local_components is not necessary if single device, so this will clutter things a bit. Try to keep things clean. Make it easy to understand and extend. Explain why this exists.  Maybe _get_local_components could be reimagined?
5. MetricsAggregator: Its a bit hard to understand "# Build global component list". Maybe make it more explicit or add some small example in the comment
6. Is it a safe decision to do "if not unique_keys:"? Why do we have it? When would that happen?
7. Why do we do device = "cpu"? Is this a good decision? Its fine to hardcode, i just want to understand. Because then you do device = torch.device(dist.get_rank()), which i also do not understand. All of the metrics are generated in the dataset, so we can expect it to be in cpu. But the collator may make it a tensor, and the recipe may send it to device. But we can deal with that later. For now, lets keep it simple.
8. _reduce_and_format_distributed is too long. Its hard to follow. The indexing is also hard to understand, e.g. k[0], k[1]. Maybe reconstruct should be another function? Maybe there is a better way to handle all that?
9. Lets avoid calling get_world_size all of the time. Do it once.
10. report[ds_name][f"{metric_name}_p05"] = data["p05"] is terrible. Lets just keep the same key for everything.
11. In state_dict, do we really need the for loop and all of the ifs? Can we just save the state dict at once? Maybe we need it only for the queus, if they are not serializable. Can you think of a good way to just use the state as close as possible, without many transformations? Too much cognitive load.
12. HfIterableDataset: I dont like the dictionary self._transforms, because metrics_transform is not applied with the others. Add a Note that the reason we dont apply them together is because it doesnt work with HF .map
13. Lets keep things simple and completely remove the data error handling and filtering. Lets delete _filter_failed_transforms. I actually deleted it. You should check if there is any tests or anywhere in the code that was checking that.
14. I think it should be the responsability of the parent dataset to do ds.dataset_name: ds.state_dict(). The dataset should NOT return {self.dataset_name: state}. Lets update the definitions so that the datasets return only their states, and not a name: dictionary
15. for InterleavedDataset, it seems that we do not need self._seed, since we apply it in the __init__ only. Do you agree?
16. As I am going through the classes, i am seeing dead/useless/old debuggin variables. I am deleting them, but if you see more, please point them out for deletion.
17. InterleavedDataset: Can we make it safer to load datasets? Maybe  self._datasets could be turned into a dictionary name: ds?
18. Collate fn is just too long. I dont think that we need all the special handling. We are not here testing collate. Just keep it as simple as possible. If necessary change create_test_json_file to be simpler. As it is, it doesnt add value
19. run_training_loop is an ok abstraction. We can keep it. But I dont like that we have to pass an iterator instead of a dataloader. I also dont like that for resume from ckpt we call it twice. I am thinking that instead we should just have a param to save checkpoint and run other N steps.
20. assert_sample_structure is used on a single test. I dont see why we need an utility for it. If anything, make it local, but i dont think its required. Like i said, the collator is not a core part of the implementation. We just added it because of the metrics.
21. Lets delete assert_checkpoint_continuation.
22. StandardMetricTransform should not have dataset_name as an input. It would be too annoying to do it via config. Lets make it a responsability of the dataset to call .set_dataset_name, or something like that. Maybe it could be a property? 
23. StandardMetricTransform: Add a docstring with the metrics being recorded and why they are recorded that way, which is to work with dataloader num_workers>0 and world_size>1. Keep it short, but insightful.
24. I am ok with keeping it a factory, but i feel that people usually dont like factories. Is there a better alternative? its fine if the answer is no.25. TestHFIterableDataset.test_epoch_boundaries: Lets just do a single dataset, use small for all, and just test the number of epochs. The small/medium/large are usefull only for multidataset testing. Keep it simple. However, the test should be more meaningful. I think it should check 1,2,3 epochs and confirm that all items are appears N epoch times. I would prefer if this was a distributed test with num_workers>126. test_shuffling_behavior you can use the small dataset. Make sure that for shuffle=True has different samples for epoch 1 and epoch 2, i.e. its shuffling on each epoch. Lets remove parametrizations and just for ds1, ds2, with 1 has shuffle and 2 doesnt have shuffle. Also assert that all ids are the same for one epoch.27. delete test_transform_error_handling, as mentioned previously28. TestHFIterableDataset.test_checkpointing can just be "mid_epoch", "epoch_boundary", "multi_epoch", but instead of this, just put "num_epochs"=0.5, 1.0, 2.5 and make it a float. It should avoid having to do all of the if/else.29. TestInterleavedDataset.test_initialization_validation should confirm that wieghts sum to 1 after normalization30. I am thinking that every test should just use small_dataset_size, unless its necessary to use something else.31. In TestHFIterableDataset lets make sure we test the dataset_name default given the path, and that its working as expected.32. test_metrics_aggregation lets make sure that we have a situation that 1 dataset has epoched and the other hasnt, and metrics capture that.?33. test_sampling_ratios simplify it. Dont have the for loop to create datasets. Just do ds1 and ds2, first small, second medium. Then check if their IDs mean are on tens and hundreds, respectively. Add a note explaining the logic.34. TestEndToEndCheckpointing has good tests, but its too long and complex. I see two options:a. Completely drop is and duplicate previous tests when distributed could alter the behavior of the test, e.g. metric reduce.b. Keep this unit test, but simplify it. Try with justinterleaved, shuffle=False, num_workers=0,3, and parametrize on number of epochs 0.5, 1, 2.5. If num_workers>0, then dont check resume from ckpt. But this still looks a bit too much. I would like to see in your plan both options to make a decision. For a) give me a list of the tests that would need to be added. For b) wirte the unit test considering all of my previous feedback. Edit: it seems that a large portion of (a) is TestDistributedDataLoading. This is highly duplicative. Please provide guidance of what we should keep/delete. I am leaning towards B and deleting TestDistributedDataLoading, but maybe have some specialized functions for distributed, like test_distributed_aggregation35. TestDistributedMetricsAggregator.setup. I dont fully follow why we need self.tmp_dir = tempfile.TemporaryDirectory(prefix=f"rank_{self.rank}_"). I dont know if there was a bug, or we added it there for no reason and never tested it. Which race condition exactly? the dataset is static.36. test_distributed_aggregation seems to have some blind spots compared to the single device, which tests all metrics.37. I dont understand why the classes have their own dataset_factory, if thats already a utility. It looks like we could delete it.38. I think we could delete TestDistributedEdgeCases

IMPORTANT: WHEN ADDING COMMENTS AND DOCSTRINGS, MAKE SURE THAT YOU SOUND HUMAN, AND NOT LIKE AN LLM.
---

### `MetricsAggregator` Refactoring

**1. Unclear Documentation**
*   **Chosen Approach (Opus & Gemini):** I will add a comprehensive docstring to the `MetricsAggregator` class and a comment for the `_state` variable, including the requirement that output should be wandb-ready.

    ```python
    class MetricsAggregator:
        """
        Aggregates metrics across datasets and distributed ranks.

        The internal state `_state` is a dictionary where the key is a tuple
        of `(dataset_name, metric_name)` and the value is another dictionary
        holding the metric's specific state (e.g., `{'type': AggregationType.SUM, 'value': 10}`).
        
        Usage:
            aggregator = MetricsAggregator()
            aggregator.update(metrics)
            # Get wandb-ready metrics
            metrics = aggregator.prepare_for_logging(prefix="train")  # {"train/dataset1/tokens": 1234, ...}
            wandb.log(metrics)
        """
        def __init__(self, dist_window_size: int = 1000):
            # State shape: {(dataset_name, metric_name): {type: AggType, value/sum/counts/etc}}
            self._state: Dict[Tuple[str, str], Dict[str, Any]] = {}
            self._dist_window_size = dist_window_size
            # ...
    ```

**2. Redundant `has_value` flag**
*   **Chosen Approach (Unanimous):** I will remove the `state["has_value"]` flag and use a `state["value"] is not None` check for `MAX` and `MIN` aggregation types.

    ```python
    # Before
    if state["has_value"]:
        state["value"] = max(state["value"], metric.value)
    else:
        state["value"] = metric.value
        state["has_value"] = True

    # After
    if state["value"] is not None:
        state["value"] = max(state["value"], metric.value)
    else:
        state["value"] = metric.value
    ```

**3. Unclear Sorting Logic for Distributions**
*   **Chosen Approach (Gemini):** I will add a comment explaining the sorting logic.

    ```python
    # Sort to get percentiles efficiently
    sorted_values = sorted(values)
    ```

**4. Merge `_compute_local_metrics` and `_get_local_components`**
*   **Chosen Approach:** Simplify the entire metrics pipeline to always produce flat, wandb-ready dictionaries with keys in the format `"{prefix}/{dataset_name}/{metric_name}"`.

    ```python
    class MetricsAggregator:
        # ... __init__, update, _initialize_state ...

        def prepare_for_logging(self, prefix: str = "") -> Dict[str, float]:
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
            Compute metrics from current state. Returns flat structure with values and their types.
            
            Returns:
                Dictionary mapping (dataset_name, metric_name) -> {"value": value, "agg_type": aggregation_type}
                For distributions and categoricals, expands into multiple entries.
                The dict format allows future extensions with additional fields.
            """
            metrics = {}
            
            for (ds_name, metric_name), state in self._state.items():
                agg_type = state["type"]
                
                if agg_type == AggregationType.SUM:
                    metrics[(ds_name, metric_name)] = {"value": state["value"], "agg_type": agg_type}
                    
                elif agg_type in (AggregationType.MAX, AggregationType.MIN):
                    if state["value"] is not None:
                        metrics[(ds_name, metric_name)] = {"value": state["value"], "agg_type": agg_type}
                        
                elif agg_type == AggregationType.MEAN:
                    if state["count"] > 0:
                        value = state["sum"] / state["count"]
                        metrics[(ds_name, metric_name)] = {"value": value, "agg_type": agg_type}
                    
                elif agg_type == AggregationType.DISTRIBUTION:
                    # Expand distribution into individual metrics
                    if state["values"]:
                        values = list(state["values"])
                        sorted_values = sorted(values)
                        n = len(sorted_values)
                        
                        # Each stat becomes its own metric
                        # For percentiles, it is an approximattion by computing avg of averages
                        metrics[(ds_name, f"{metric_name}_mean")] = {"value": sum(values) / n, "agg_type": AggregationType.MEAN}
                        metrics[(ds_name, f"{metric_name}_min")] = {"value": sorted_values[0], "agg_type": AggregationType.MIN}
                        metrics[(ds_name, f"{metric_name}_max")] = {"value": sorted_values[-1], "agg_type": AggregationType.MAX}
                        metrics[(ds_name, f"{metric_name}_p05")] = {"value": sorted_values[max(0, int(0.05 * n) - 1)], "agg_type": AggregationType.MEAN}
                        metrics[(ds_name, f"{metric_name}_p50")] = {"value": sorted_values[max(0, int(0.5 * n) - 1)], "agg_type": AggregationType.MEAN}
                        metrics[(ds_name, f"{metric_name}_p95")] = {"value": sorted_values[max(0, int(0.95 * n) - 1)], "agg_type": AggregationType.MEAN}
                        
                elif agg_type == AggregationType.CATEGORICAL_COUNT:
                    # Expand categorical counts into individual metrics
                    for category, count in state["counts"].items():
                        metrics[(ds_name, f"{metric_name}_{category}_count")] = {"value": count, "agg_type": AggregationType.SUM}
                        
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
            """
            world_size = dist.get_world_size()
            
            # Gather all metrics from all ranks in one operation
            all_metrics = [None] * world_size
            dist.all_gather_object(all_metrics, local_metrics)
            
            # Group values by key for reduction
            grouped = collections.defaultdict(list)
            for rank_metrics in all_metrics:
                for key, metric_dict in rank_metrics.items():
                    grouped[key].append(metric_dict)
            
            # Reduce based on aggregation type
            reduced = {}
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
            self, 
            metrics: Dict[Tuple[str, str], Dict[str, Any]], 
            prefix: str
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
    ```

**5-10. Additional Cleanups**
*   **Note:** These points are now addressed by the simplified implementation in section 4:
    - No more complex component lists or tensor operations
    - No device selection needed (all operations on Python objects)
    - `_reduce_and_format_distributed` is replaced by simpler `_compute_distributed_metrics`
    - `get_world_size()` is called once per method that needs it
    - Distribution metrics are expanded in `_compute_local_metrics` with proper naming

**11. Simplify `MetricsAggregator.state_dict`**
*   **Chosen Approach (Gemini & Opus):** The implementation will directly serialize the state, only converting `deque` and `Counter` to their list/dict representations. Keys will be converted to strings for JSON compatibility.

    ```python
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
            # Using ast.literal_eval for security (only evaluates literals, not arbitrary code)
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
    ```

---

### Dataset and General Refactoring

**12. Clarify Transform Application in `HFIterableDataset`**
*   **Chosen Approach (Gemini & Opus):** I will use separate, explicit members for each transform and add a clarifying comment.

    ```python
    class HFIterableDataset(TuneIterableDataset):
        def __init__(self, ..., message_transform, model_transform, output_transform, metric_transform):
            self._message_transform = message_transform or (lambda x: x)
            self._model_transform = model_transform or (lambda x: x)
            self._output_transform = output_transform or (lambda x: x)
            self._metric_transform = metric_transform or (lambda x: x)
            # ...

        def __iter__(self) -> Iterator[Dict[str, Any]]:
            # ...
            for sample in epoch_iterator:
                # Note: We apply transforms here instead of in a single .map() call
                # to work around https://github.com/huggingface/datasets/issues/7630
                # where .map() can cause incorrect resumption from a checkpoint.
                sample = self._message_transform(sample)
                sample = self._model_transform(sample)
                sample = self._output_transform(sample)
                sample = self._metric_transform(sample)
                # ...
                yield sample
    ```

**13. Remove Error Handling and Filtering**
*   **Chosen Approach (Unanimous):** All code related to `_filter_failed_transforms` and `max_transform_failures_per_epoch` will be removed. Specifically:
    - Remove `max_transform_failures_per_epoch` parameter from `__init__`
    - Remove `self._transform_failures_this_epoch` instance variable
    - Remove `self._max_transform_failures_per_epoch` instance variable
    - Remove any try/except blocks around transform application
    - Remove the `_filter_failed_transforms` method entirely
    - Remove `transform_failures_this_epoch` from state_dict/load_state_dict

**14. Simplify `state_dict` for `TuneIterableDataset`**
*   **Chosen Approach (Sonnet):** Datasets will return their state dictionary directly. The parent (`InterleavedDataset`) will be responsible for namespacing.

    ```python
    # In HFIterableDataset
    def state_dict(self) -> Dict[str, Any]:
        # The dataset returns its own state directly, without namespacing.
        return {
            "num_epochs": self._num_epochs,
            "seed": self._seed,
            "hf_dataset_state": self._ds.state_dict(),
        }

    # In InterleavedDataset
    def state_dict(self) -> Dict[str, Any]:
        # The parent is responsible for namespacing the child states.
        child_states = {
            name: ds.state_dict() for name, ds in self._datasets.items()
        }
        return {
            "sampling_generator_state": self._sampling_generator.get_state(),
            "child_states": child_states,
        }

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        self._sampling_generator.set_state(state_dict["sampling_generator_state"])
        child_states = state_dict["child_states"]
        for name, ds in self._datasets.items():
            if name in child_states:
                # Pass the raw state dict to the child
                ds.load_state_dict(child_states[name])
    ```

**15. Remove Unused `_seed` in `InterleavedDataset`**
*   **Chosen Approach (Unanimous):** The `self._seed` attribute will be removed.

**16. Remove Dead Code**
*   **Chosen Approach (Unanimous):** I will scan the file and remove any unused variables. Specifically:
    - Any debug print statements or logging that was commented out
    - Unused imports
    - Variables that are assigned but never used
    - Dead code branches that can never be reached
    - Old commented-out implementations

**17. Improve `InterleavedDataset` Safety**
*   **Chosen Approach (Sonnet):** I will change `self._datasets` to a dictionary and store an ordered list of names for sampling.

    ```python
    class InterleavedDataset(TuneIterableDataset):
        def __init__(self, datasets: List[TuneIterableDataset], ...):
            # Create a name-to-dataset mapping for robust state management
            self._datasets: Dict[str, TuneIterableDataset] = {
                ds.dataset_name: ds for ds in datasets
            }
            # Preserve original order for weighted sampling
            self._dataset_names = [ds.dataset_name for ds in datasets]
            # ...

        def __iter__(self) -> Iterator[Dict[str, Any]]:
            child_iters = {name: iter(ds) for name, ds in self._datasets.items()}
            while True:
                # Sample an index, then get the name for safe lookup
                ds_idx = torch.multinomial(...).item()
                ds_name = self._dataset_names[ds_idx]
                yield next(child_iters[ds_name])
                # ... with StopIteration handling ...
    ```

**18. Simplify `collate_with_metrics`**
*   **Chosen Approach (Opus):** I will simplify the collate function.

    ```python
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
        padding_value=-1  # Use -1 for padding to distinguish from valid IDs
    )

    return {
        "id": ids,
        "tokens": tokens,
        "metrics": all_metrics,
    }
    ```

**19. Refactor `run_training_loop`**
*   **Chosen Approach:** Create a test utility function that simplifies checkpoint testing. It will be placed in the test file as a helper function, not a fixture.

    ```python
    # Test utility for checkpoint testing
    def generate_ckpt(
        dataloader: DataLoader,  # Just pass the dataloader directly
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
                metrics_at_checkpoint = aggregator.prepare_for_logging(prefix="train")
            
            # Stop after total steps
            if idx == steps_before_checkpoint + steps_after_checkpoint - 1:
                break

        # Split batches
        pre_checkpoint_batches = batches[:steps_before_checkpoint]
        post_checkpoint_batches = batches[steps_before_checkpoint:]
        
        # Resume with new instances if provided
        resumed_batches = []
        resumed_metrics = {}
        
        if resume_dataloader is not None and resume_aggregator is not None:
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
            
            resumed_metrics = resume_aggregator.prepare_for_logging(prefix="train")
        
        return {
            # Original run
            "pre_checkpoint_batches": pre_checkpoint_batches,
            "post_checkpoint_batches": post_checkpoint_batches,
            "metrics_at_checkpoint": metrics_at_checkpoint,
            "metrics": aggregator.prepare_for_logging(prefix="train"),
            
            # Resumed run  
            "resumed_batches": resumed_batches, 
            "resumed_metrics": resumed_metrics,
            
            # Internal state for loading - only if someone needs to manually load
            "_checkpoint_state": checkpoint_state,
        }
    
    # Usage in tests would be simpler:
    def test_checkpointing_example(self):
        # Create dataset and dataloader
        dataset = create_dataset()
        dataloader = StatefulDataLoader(dataset, ...)
        aggregator = MetricsAggregator()
        
        # Create new instances for resuming
        dataset2 = create_dataset()
        dataloader2 = StatefulDataLoader(dataset2, ...)
        aggregator2 = MetricsAggregator()
        
        # Run with checkpoint and automatic resume testing
        result = generate_ckpt(
            dataloader, aggregator, 
            steps_before_checkpoint=10,
            steps_after_checkpoint=5,
            resume_dataloader=dataloader2,
            resume_aggregator=aggregator2
        )
        
        # Verify results
        assert len(result["pre_checkpoint_batches"]) == 10
        assert len(result["post_checkpoint_batches"]) == 5
        assert len(result["resumed_batches"]) == 5  # Should match post_checkpoint_batches
        
        # Metrics should show progression
        checkpoint_metrics = result["metrics_at_checkpoint"]
        final_metrics = result["metrics"]
        resumed_metrics = result["resumed_metrics"]
        
        # Original run should have processed all samples
        assert final_metrics["train/dataset/samples_seen"] > checkpoint_metrics["train/dataset/samples_seen"]
        
        # Resumed run should match the original final metrics
        assert resumed_metrics == final_metrics
    
    # Tests in debug_iterable_dataset.py that should use this utility:
    # - TestHFIterableDataset.test_checkpointing - for basic dataset checkpointing
    # - TestInterleavedDataset.test_checkpointing - for interleaved dataset state  
    # - TestDistributedEndToEnd.test_distributed_checkpoint_metrics (already updated above)
    # 
    # Note: We cannot test checkpoint/resume with num_workers > 0 or shuffle=True
    # because StatefulDataLoader doesn't guarantee exact reproducibility in those cases.
    # Instead, we test that:
    # 1. Each epoch contains the same set of samples (but potentially different order)
    # 2. No duplicates within an epoch
    # 
    # Benefits of using generate_ckpt:
    # - Handles the checkpoint/resume logic in one place
    # - Automatically tests both continuing with same instances and resuming with new ones
    # - Returns structured data for easy verification
    # - Reduces code duplication across tests
    ```

**20-21. Remove `assert_sample_structure` and `assert_checkpoint_continuation`**
*   **Chosen Approach (Unanimous):** These test utilities will be removed.

**22-23. Refactor `StandardMetricTransform`**
*   **Chosen Approach:** Use a method `set_dataset_name` rather than a property for clarity and explicit initialization. Create a proper `MetricTransform` protocol/base class.

    ```python
    class MetricTransform(Protocol):
        """Protocol for metric transforms."""
        def set_dataset_name(self, dataset_name: str) -> None: ...
        def __call__(self, sample: Dict[str, Any]) -> Dict[str, Any]: ...
    
    class StandardMetricTransform:
        """
        Adds per-sample metrics for tracking training progress.

        It is not a responsability of this class to handle the aggregation of metrics.
        If you want to get cumulative sum of samples_seen, for example, you should
        add Metric(dataset_name=ds_name, name="samples_seen", value=1, agg_type=AggregationType.SUM)
        to the sample["metrics"] list.
        
        This is designed this way to ensure correct aggregation
        with multiple dataloader workers and distributed training.
        
        Tracks:
        - samples_seen: count of samples processed  
        - tokens_seen: total tokens processed
        - seq_len: distribution of sequence lengths
        
       
        """
        def __init__(self):
            self.dataset_name: Optional[str] = None
            self.new_metric: Optional[Callable] = None

        def set_dataset_name(self, dataset_name: str) -> None:
            """Called by dataset to set the namespace for metrics."""
            self.dataset_name = dataset_name
            self.new_metric = partial(Metric, dataset_name=dataset_name)

        def __call__(self, sample: Dict[str, Any]) -> Dict[str, Any]:
            if self.dataset_name is None or self.new_metric is None:
                raise RuntimeError("set_dataset_name() must be called before using the transform.")
            
            # Determine token key (support both 'tokens' and 'input_ids')
            token_key = "tokens" if "tokens" in sample else "input_ids"
            token_len = len(sample.get(token_key, []))

            # Create metrics for this sample
            metrics = [
                self.new_metric(name="samples_seen", value=1, agg_type=AggregationType.SUM),
                self.new_metric(name="tokens_seen", value=token_len, agg_type=AggregationType.SUM),
                self.new_metric(name="seq_len", value=token_len, agg_type=AggregationType.DISTRIBUTION),
            ]

            # Append to existing metrics list or create new one
            if "metrics" not in sample:
                sample["metrics"] = []
            sample["metrics"].extend(metrics)
            return sample

    # Usage in HFIterableDataset
    class HFIterableDataset(TuneIterableDataset):
        def __init__(self, ..., metric_transform: Optional[Callable] = None):
            # ... other initialization ...
            
            # Create default transform if not provided
            self._metric_transform = metric_transform or StandardMetricTransform()
            
            # Set dataset name on the transform if it supports it
            if hasattr(self._metric_transform, 'set_dataset_name'):
                self._metric_transform.set_dataset_name(self.dataset_name)
    ```
*   **Design Notes:**
    - Method over property: More explicit and avoids timing issues during initialization
    - Protocol allows other metric transforms to follow the same pattern
    - Type hints are clear: `value: int` for SUM metrics, `value: int` for DISTRIBUTION (sequence length)

**24. Alternative to `dataset_factory`**
*   **Chosen Approach (Unanimous):** I will keep the `dataset_factory` pytest fixture.

---

### Test Suite Refactoring

**25-33. Test Simplification**
*   **Chosen Approach (Synthesized from Sonnet):**
    *   **`test_epoch_boundaries`:** Will be simplified to verify that for N epochs, each sample appears exactly N times in a non-shuffled dataset.
    *   **`test_shuffling_behavior`:** Will use one shuffled and one unshuffled dataset to confirm that shuffling changes order between epochs while preserving the set of samples.
    *   **`test_sampling_ratios`:** Will use two datasets with distinct ID ranges (e.g., 0-99 and 1000-1099) to make verifying the sampling ratio trivial by checking the range of IDs in the output.

**34. Refactor `TestEndToEndCheckpointing` and `TestDistributedDataLoading`**
*   **Chosen Approach:** Keep a simplified end-to-end test focused on distributed checkpointing. Delete `TestDistributedDataLoading` as it's duplicative.

    ```python
    class TestDistributedEndToEnd(FSDPTest):
        @property
        def world_size(self) -> int:
            return 2
        
        def setUp(self):
            super().setUp()
            # Rank-specific temp dir to avoid file system race conditions
            self.tmp_dir = tempfile.TemporaryDirectory(prefix=f"rank_{self.rank}_")
            self.tmp_path = Path(self.tmp_dir.name)
        
        def tearDown(self):
            self.tmp_dir.cleanup()
            super().tearDown()

        @gpu_test(gpu_count=2)
        @pytest.mark.parametrize("num_epochs", [0.5, 1.0, 2.5])
        def test_distributed_checkpoint_metrics(self, num_epochs):
            """
            Test that metrics are correctly preserved across checkpoint/resume
            in a distributed setting with interleaved datasets.
            
            What's being tested:
            • Distributed metrics aggregation works correctly
            • Checkpoint/resume preserves metric state in distributed setting
            • Interleaved dataset state is properly saved/restored
            • Metrics continue accumulating correctly after resume
            """
            def create_dataset():
                # Create interleaved dataset with no shuffle for determinism
                file1 = self.tmp_path / "ds1.json"
                file2 = self.tmp_path / "ds2.json"
                create_test_json_file(file1, SMALL_DATASET_SIZE)
                create_test_json_file(file2, SMALL_DATASET_SIZE, offset=100)
                
                ds1 = HFIterableDataset(
                    path="json",
                    data_files=str(file1),
                    split="train",
                    dataset_name="ds1",
                    shuffle_buffer_size=0,  # No shuffle for determinism
                    metric_transform=StandardMetricTransform(),
                )
                ds2 = HFIterableDataset(
                    path="json",
                    data_files=str(file2),
                    split="train",
                    dataset_name="ds2",
                    shuffle_buffer_size=0,  # No shuffle for determinism
                    metric_transform=StandardMetricTransform(),
                )
                
                return InterleavedDataset([ds1, ds2], [0.7, 0.3], seed=SEED)
            
            def create_dataloader(dataset):
                # num_workers=0 for checkpoint testing
                loader = StatefulDataLoader(
                    dataset,
                    batch_size=BATCH_SIZE,
                    num_workers=0,  # Required for determinism
                    collate_fn=collate_with_metrics,
                )
                return loader, MetricsAggregator()
            
            # Calculate steps
            total_samples = int(SMALL_DATASET_SIZE * 2 * num_epochs)
            total_steps = total_samples // BATCH_SIZE
            steps_before = total_steps // 2
            steps_after = total_steps - steps_before
            
            # Run with checkpoint and resume
            loader1, aggregator1 = create_dataloader(create_dataset())
            loader2, aggregator2 = create_dataloader(create_dataset())
            
            result = generate_ckpt(
                loader1, aggregator1,
                steps_before, steps_after,
                resume_dataloader=loader2,
                resume_aggregator=aggregator2
            )
            
            # Get metrics
            checkpoint_metrics = result["metrics_at_checkpoint"]
            final_metrics = result["metrics"]
            resumed_metrics = result["resumed_metrics"]
            
            # Verify both datasets contributed and check specific values
            # With new format, metrics are flat: "train/ds1/samples_seen"
            assert "train/ds1/samples_seen" in final_metrics
            assert "train/ds2/samples_seen" in final_metrics
            
            # Check samples_seen values
            ds1_samples = final_metrics["train/ds1/samples_seen"]
            ds2_samples = final_metrics["train/ds2/samples_seen"]
            total_samples_seen = ds1_samples + ds2_samples
            
            # Should match expected total
            assert total_samples_seen == total_samples, (
                f"Expected {total_samples} samples, got {total_samples_seen}"
            )
            
            # ds1 should have ~70% of samples
            ratio = ds1_samples / total_samples_seen
            assert 0.6 < ratio < 0.8, f"Unexpected sampling ratio: {ratio}"
            
            # Check distribution metrics exist and have reasonable values
            assert "train/ds1/seq_len_mean" in final_metrics
            assert "train/ds1/seq_len_min" in final_metrics
            assert "train/ds1/seq_len_max" in final_metrics
            assert "train/ds1/seq_len_p50" in final_metrics
            
            # Since we control the data generation, we know seq lengths are 1-3
            assert 1 <= final_metrics["train/ds1/seq_len_min"] <= 3
            assert 1 <= final_metrics["train/ds1/seq_len_max"] <= 3
            assert 1 <= final_metrics["train/ds1/seq_len_mean"] <= 3
            assert 1 <= final_metrics["train/ds1/seq_len_p50"] <= 3
            
            # Verify metrics increased from checkpoint to final
            assert final_metrics["train/ds1/samples_seen"] > checkpoint_metrics["train/ds1/samples_seen"]
            assert final_metrics["train/ds2/samples_seen"] > checkpoint_metrics["train/ds2/samples_seen"]

        @gpu_test(gpu_count=2)
        def test_distributed_shuffle_epoch_consistency(self):
            """
            Test that with shuffle=True and num_workers=3, each epoch contains
            the same set of samples (but in different order).
            
            What's being tested:
            • Shuffling changes order between epochs but preserves all samples
            • Each epoch contains exactly the same set of sample IDs  
            • Multiple workers don't cause duplicate or missing samples
            • Distributed sharding works correctly with shuffling
            """
            def create_dataset():
                # Use a single dataset for clearer epoch tracking
                file1 = self.tmp_path / "shuffle_test.json"
                create_test_json_file(file1, SMALL_DATASET_SIZE)
                
                return HFIterableDataset(
                    path="json",
                    data_files=str(file1),
                    split="train",
                    dataset_name="shuffle_test",
                    shuffle_buffer_size=DEFAULT_SHUFFLE_BUFFER_SIZE,  # Shuffle enabled
                    metric_transform=StandardMetricTransform(),
                )
            
            dataset = create_dataset()
            loader = StatefulDataLoader(
                dataset,
                batch_size=BATCH_SIZE,
                num_workers=3,  # Multiple workers as requested
                collate_fn=collate_with_metrics,
            )
            
            # Run for 2 complete epochs
            samples_per_epoch = SMALL_DATASET_SIZE
            batches_per_epoch = (samples_per_epoch + BATCH_SIZE - 1) // BATCH_SIZE
            total_batches = batches_per_epoch * 2
            
            # Collect IDs for each epoch
            epoch1_ids = []
            epoch2_ids = []
            
            iterator = iter(loader)
            
            # Epoch 1
            for _ in range(batches_per_epoch):
                batch = next(iterator)
                epoch1_ids.extend(batch["id"].tolist())
            
            # Epoch 2
            for _ in range(batches_per_epoch):
                batch = next(iterator)
                epoch2_ids.extend(batch["id"].tolist())
            
            # Verify both epochs have the expected number of samples  
            # Note: With multiple workers, we may have slight variations due to sharding
            assert abs(len(epoch1_ids) - samples_per_epoch) <= self.world_size, (
                f"Epoch 1 has {len(epoch1_ids)} samples, expected ~{samples_per_epoch}"
            )
            assert abs(len(epoch2_ids) - samples_per_epoch) <= self.world_size, (
                f"Epoch 2 has {len(epoch2_ids)} samples, expected ~{samples_per_epoch}"
            )
            
            # Verify all samples are unique within each epoch
            assert len(set(epoch1_ids)) == len(epoch1_ids), "Epoch 1 has duplicate samples"
            assert len(set(epoch2_ids)) == len(epoch2_ids), "Epoch 2 has duplicate samples"
            
            # Verify both epochs contain the same set of samples
            assert set(epoch1_ids) == set(epoch2_ids), (
                "Epochs should contain the same samples.\n"
                f"Epoch 1 only: {set(epoch1_ids) - set(epoch2_ids)}\n"
                f"Epoch 2 only: {set(epoch2_ids) - set(epoch1_ids)}"
            )
            
            # Verify order is different (due to shuffle)
            assert epoch1_ids != epoch2_ids, "Shuffled epochs should have different order"
            
            # Verify we saw all expected IDs
            expected_ids = set(range(SMALL_DATASET_SIZE))
            assert set(epoch1_ids) == expected_ids, (
                f"Missing IDs: {expected_ids - set(epoch1_ids)}"
            )
    ```

*   **Key Changes:**
    - Removed shuffle for deterministic testing
    - Set num_workers=0 for checkpoint compatibility
    - Uses the `generate_ckpt` utility
    - Focused on metrics consistency rather than sample-level comparison
    - Added separate test for all aggregation types in distributed setting
    - Deleted `TestDistributedDataLoading` to avoid duplication

**35. Distributed Test Setup**
*   **Chosen Approach (Gemini):** I will keep the rank-specific temporary directory and add a comment explaining it's a defensive measure against potential race conditions on shared file systems.

**36. Expand `test_distributed_aggregation` Coverage**
*   **Chosen Approach:** Here's a comprehensive test for all aggregation types in distributed setting:

    ```python
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
                Metric("test", "dist_metric", rank * 10 + i, AggregationType.DISTRIBUTION)
            )
        
        # CATEGORICAL_COUNT: Different categories per rank
        # rank 0: 3 of cat_A, 2 of cat_B
        # rank 1: 1 of cat_A, 4 of cat_C
        if rank == 0:
            metrics.extend([
                Metric("test", "cat_metric", "cat_A", AggregationType.CATEGORICAL_COUNT),
                Metric("test", "cat_metric", "cat_A", AggregationType.CATEGORICAL_COUNT),
                Metric("test", "cat_metric", "cat_A", AggregationType.CATEGORICAL_COUNT),
                Metric("test", "cat_metric", "cat_B", AggregationType.CATEGORICAL_COUNT),
                Metric("test", "cat_metric", "cat_B", AggregationType.CATEGORICAL_COUNT),
            ])
        else:
            metrics.extend([
                Metric("test", "cat_metric", "cat_A", AggregationType.CATEGORICAL_COUNT),
                Metric("test", "cat_metric", "cat_C", AggregationType.CATEGORICAL_COUNT),
                Metric("test", "cat_metric", "cat_C", AggregationType.CATEGORICAL_COUNT),
                Metric("test", "cat_metric", "cat_C", AggregationType.CATEGORICAL_COUNT),
                Metric("test", "cat_metric", "cat_C", AggregationType.CATEGORICAL_COUNT),
            ])
        
        # Update aggregator
        aggregator.update(metrics)
        
        # Get distributed metrics
        result = aggregator.prepare_for_logging(prefix="train")
        
        # Verify SUM aggregation
        assert result["train/test/sum_metric"] == 30, (
            f"Expected sum 30, got {result['train/test/sum_metric']}"
        )
        
        # Verify MEAN aggregation  
        assert result["train/test/mean_metric"] == 20, (
            f"Expected mean 20, got {result['train/test/mean_metric']}"
        )
        
        # Verify MAX aggregation
        assert result["train/test/max_metric"] == 200, (
            f"Expected max 200, got {result['train/test/max_metric']}"
        )
        
        # Verify MIN aggregation
        assert result["train/test/min_metric"] == 5, (
            f"Expected min 5, got {result['train/test/min_metric']}"
        )
        
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
        assert result["train/test/dist_metric_mean"] == 7, (
            f"Expected mean 7, got {result['train/test/dist_metric_mean']}"
        )
        
        # Verify CATEGORICAL_COUNT aggregation
        # Total: cat_A: 4, cat_B: 2, cat_C: 4
        assert result["train/test/cat_metric_cat_A_count"] == 4, (
            f"Expected cat_A count 4, got {result['train/test/cat_metric_cat_A_count']}"
        )
        assert result["train/test/cat_metric_cat_B_count"] == 2, (
            f"Expected cat_B count 2, got {result['train/test/cat_metric_cat_B_count']}"
        )
        assert result["train/test/cat_metric_cat_C_count"] == 4, (
            f"Expected cat_C count 4, got {result['train/test/cat_metric_cat_C_count']}"
        )
        
        # Test edge case: metrics from only one rank
        aggregator_single = MetricsAggregator()
        if rank == 0:
            aggregator_single.update([
                Metric("single", "only_rank0", 42, AggregationType.SUM),
            ])
        
        result_single = aggregator_single.prepare_for_logging(prefix="train")
        
        # Should still work even if only one rank has the metric
        assert result_single["train/single/only_rank0"] == 42, (
            f"Single rank metric failed: {result_single.get('train/single/only_rank0')}"
        )
    ```

**37. Consolidate `dataset_factory` Usage**
*   **Chosen Approach (Unanimous):** I will remove any duplicate, class-level `dataset_factory` methods and use the single pytest fixture consistently.

**38. Delete `TestEdgeCases`**
*   **Chosen Approach (Unanimous):** This test class and its contents will be removed.

---

### Questions and Concerns Addressed

This section clarifies design choices and addresses the questions and potential blind spots raised in the initial planning documents, explaining how the final plan resolves them.

*   **Error Propagation:** Per your request, error handling in data transforms has been removed. Transform errors will now correctly cause a crash, which is a cleaner design than silently swallowing potential data corruption issues.

*   **Distributed Percentile Calculation:** The distributed calculation for percentiles is an **approximation**, a design choice favoring performance over perfect accuracy. We do not gather all sequence lengths from all ranks (which would be prohibitively expensive). Instead, each rank calculates its local statistics (min, max, p05, p50, p95), and these values are then reduced across ranks (min of mins, max of maxes, average of means/percentiles). This provides a reasonable and lightweight estimate suitable for logging.

*   **Interleaved Dataset Epoch Tracking:** The refactored `InterleavedDataset` checkpoints the state of each child dataset independently, including the child's internal epoch count. The new `test_distributed_checkpoint_metrics_consistency` test will implicitly validate this behavior, as `num_epochs` is a logged metric that is tracked per-dataset. For the final aggregated metrics to be identical between a continuous and a resumed run, this state must be saved and restored correctly.

*   **Performance of Categorical & Distribution Metrics:** The current implementation for categorical metrics (`all_gather_object`) and distribution metrics (`collections.deque`) is designed for simplicity and correctness. Performance optimizations for extreme edge cases (e.g., millions of unique categories) are considered out of scope for this refactoring. The `dist_window_size` for distributions is a configurable parameter to allow users to manage memory consumption.

*   **Checkpoint Compatibility:** The changes to `state_dict` formats are breaking. Given this is a development and testing script (`debug_iterable_dataset.py`), backward compatibility for checkpoints is not a requirement for this refactor.

*   **Device Management:** The plan simplifies metric reduction to be CPU-based. This makes it universally compatible with all distributed backends (e.g., `gloo`, `nccl`) and removes device management complexity, which was a source of bugs in the original script.

*   **Robust Metric Type Detection:** The new implementation expands all metrics (distributions, categoricals) into individual key-value pairs in `_compute_local_metrics`. Each metric carries its aggregation type alongside its value, eliminating the need for complex type detection during distributed reduction. This provides a clean, simple, and extensible approach that produces wandb-ready output directly.