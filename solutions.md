# TorchTune Debugging Log

This document tracks the attempts to fix issues related to `LinearCrossEntropyLoss` in `torchtune`.

## Problem Statement

The goal is to have a memory-efficient `LinearCrossEntropyLoss` that works correctly with `torch.compile` and tensor-parallel distributed training (`DTensor`).

## Old logic issue

## Attempt 1: The Original `OLDLinearCrossEntropyLoss`

The setup to the problem is that we have bunch of tokens and a mask ([False, False, ..., True, True]) of which tokens are trainable. Instead of computing the logits for all of the tokens and then computing cross entropy loss, we instead want to mask the tokens to only compute the logits (final layer of the LLM) for the trainable tokens. This looks like:

loss = LinearCrossEntropy(model.logits_layer)
model.skip_logits_layer = True

...

for tokens, labels in dataset:
    h = model(tokens)
    l = loss(h, labels)
A simplified version of what happens inside of the loss is:

def forward(self, hidden, labels):
    mask = labels != -100
    if isinstance(hidden, DTensor):
        mesh, placements = hidden.device_mesh, hidden.placements
        local_hidden = hidden.to_local()[mask]
        hidden = DTensor.from_local(local_hidden, mesh, placements)
    else:
        hidden = hidden[mask]

    logits = self.logits_layer(hidden)
    if isinstance(logits, DTensor):
        logits = logits.full_tensor()

    return F.cross_entropy(logits.float(), labels)
This solution works with TP but fails when the loss module is compiled and it seems to fail on the isinstance check. I'm not sure how to further debug this so I'm putting up for help.

Notes:

If you remove the masking and logits_layer from the loss, the loss compiles for most models but still fails on models with tied embeddings
The error message for the above example (no tied embeddings) torch._dynamo.exc.InternalTorchDynamoError: TypeError: unhashable type: non-nested SymInt. Seems to error on isinstance.
The provided code doesn't take advantage of loss_parallel which could potentially help to get rid of the DTensor checks. The masking check could be done and then we'd only compile the logits_layer call and cross_entropy call together. The loss_parallel solution might not work for RL linear losses though.

-   **Implementation**: A version of the loss function that had issues with `torch.compile`. It performed masking by converting a `DTensor` to a local tensor, applying a boolean mask, and then converting it back to a `DTensor`.
-   **Result**: It would not compile correctly when used with `DTensor`. While no specific traceback was provided for this initial version, this pattern is known to fail with `torch.compile` because it cannot trace through `isinstance` checks or the dynamic shapes produced by `DTensor.from_local` after masking.

-   **Code Snippet:**
    ```python
    # From OLDLinearCrossEntropyLoss.compute_cross_entropy
    def compute_cross_entropy(self, hidden_chunk, target_chunk):
        # ...
        if isinstance(hidden_chunk, DTensor):
            # DTensor doesn't support masks so we have to mask locally
            mesh = hidden_chunk.device_mesh
            placements = hidden_chunk.placements
            local_hidden_chunk = hidden_chunk.to_local()[mask_chunk]
            hidden_chunk = DTensor.from_local(
                local_hidden_chunk, mesh, placements
            )
        else:
            hidden_chunk = hidden_chunk[mask_chunk]
        # ...
        logits = self.linear_projection(hidden_chunk)
        # ...
    ```

## Attempt 2: First `LinearCrossEntropyLoss` with `torch.compile`

-   **Implementation**: A new version that separated the logic to handle `DTensor` inputs by converting them to local tensors in the `forward` method. The core computation in `compute_cross_entropy` was compiled, but it called `self.linear_projection` directly on a local tensor.
-   **Result**: This failed during `torch.compile`'s "fake tensor" propagation with a shape mismatch error. `torch.compile` had trouble tracing the `forward` method of the sharded `linear_projection` layer when called with a local, non-distributed tensor.

-   **Error Log:**
    ```
    torch._dynamo.exc.TorchRuntimeError: Dynamo failed to run FX node with fake tensors: ...
    got RuntimeError('a and b must have same reduction dim, but got [8096, 32768] X [4096, 128256].')

    from user code:
    File "/data/users/felipemello/torchtune/torchtune/modules/loss/cross_entropy_loss.py", line 106, in torch_dynamo_resume_in_compute_cross_entropy_at_87
        logits = self.linear_projection(valid_hidden)
    ```

-   **Code Snippet:**
    ```python
    # From compute_cross_entropy in the first version of the new loss
    def compute_cross_entropy(self, hidden_chunk, target_chunk):
        # ... masking logic to get valid_hidden ...
        logits = self.linear_projection(valid_hidden)
        # ... compute loss ...
    ```

## Attempt 3: Refactored `LinearCrossEntropyLoss` (User's Try)

-   **Implementation**: A refactoring of the loss class to perform masking and token selection in the `forward` method and then attempting to reconstruct a `DTensor` from the local `valid_hidden` tensor before passing it to the compiled `compute_cross_entropy` method.
-   **Result**: This led to a `CUDA error: device-side assert triggered`. This error often points to a mismatch in how data is distributed across ranks, causing out-of-bounds memory access on one of the devices.

-   **Error Log:**
    ```
    /pytorch/aten/src/ATen/native/cuda/IndexKernelUtils.cu:16: vectorized_gather_kernel: ...
    Assertion `ind >=0 && ind < ind_dim_size && "vectorized gather kernel index out of bounds"` failed.
    ...
    torch.AcceleratorError: CUDA error: device-side assert triggered

    from user code:
    File "/data/users/felipemello/torchtune/torchtune/modules/loss/cross_entropy_loss.py", line 131, in forward
        valid_targets = torch.index_select(target_flat, 0, valid_indices)
    ```

-   **Code Snippet:**
    ```python
    # From the user's refactored `forward` method
    if isinstance(self.linear_projection.weight, DTensor):
        mesh = self.linear_projection.weight.device_mesh
        placements = self.linear_projection.weight.placements
        valid_hidden = DTensor.from_local(valid_hidden, mesh, placements)
    loss_sum, num_valid_tokens = self.compute_cross_entropy(valid_hidden, valid_targets)
    ```

## Attempt 4: `F.linear` with Mixed Tensor Types

-   **Implementation**: This attempt used `F.linear` with a local tensor for `valid_hidden` and a `DTensor` for the weight inside the compiled region.
-   **Result**: This led to a clear error, confirming that we cannot mix tensor types in distributed operations.

-   **Error Log:**
    ```
    torch._dynamo.exc.TorchRuntimeError: Dynamo failed to run FX node with fake tensors: ...
    got RuntimeError('aten.mm.default: got mixed torch.Tensor and DTensor, need to convert all torch.Tensor to DTensor before calling distributed operators!')

    from user code:
    File "/data/users/felipemello/torchtune/torchtune/modules/loss/cross_entropy_loss.py", line 109, in torch_dynamo_resume_in_compute_cross_entropy_at_87
        logits = F.linear(...)
    ```

-   **Code Snippet:**
    ```python
    # The F.linear call that caused the mixed-type error
    logits = F.linear(
        valid_hidden, # This is a torch.Tensor
        self.linear_projection.weight, # This is a DTensor
        self.linear_projection.bias,
    )
    ```

## Attempt 5: Refactor to Isolate DTensor Creation

-   **Implementation**: This refactoring moved the masking logic and the creation of a replicated `DTensor` from `valid_hidden` into the `forward` method (eager mode). A new, smaller `_compute_proj_and_loss` function was created to be compiled, which only performed the projection and loss calculation.
-   **Result**: This failed with the same `CUDA error: device-side assert triggered` as Attempt 3, indicating a fundamental problem with the data being passed to `torch.index_select` or a synchronization issue.

-   **Error Log:**
    ```
    /pytorch/aten/src/ATen/native/cuda/IndexKernelUtils.cu:16: vectorized_gather_kernel: ...
    Assertion `ind >=0 && ind < ind_dim_size && "vectorized gather kernel index out of bounds"` failed.
    ...
    torch.AcceleratorError: CUDA error: device-side assert triggered

    from user code:
    File "/data/users/felipemello/torchtune/torchtune/modules/loss/cross_entropy_loss.py", line 142, in forward
        valid_targets = torch.index_select(target_flat, 0, valid_indices)
    ```

-   **Code Snippet:**
    ```python
    # In `forward` method (not compiled)
    if is_dtensor_input:
        valid_hidden = DTensor.from_local(
            valid_hidden, outputs.device_mesh, [Replicate()]
        )
    loss_sum, num_valid_tokens = self._compute_proj_and_loss(valid_hidden, valid_targets)

    # In `_compute_proj_and_loss` method (compiled)
    def _compute_proj_and_loss(self, valid_hidden, valid_targets):
        # valid_hidden is now a DTensor
        logits = self.linear_projection(valid_hidden)
        # ... compute loss ...
    ```

## Attempt 6: Add Detailed Logging

-   **Implementation**: To debug the recurring CUDA error, detailed, rank-specific logging was added to the `forward` method of the implementation from Attempt 5. This was done to inspect the shapes and value ranges of the tensors on each rank right before the failing `index_select` call.
-   **Rationale**: The repeated CUDA indexing error suggests a subtle bug in the data distribution that isn't visible from the stack trace alone. The logging is intended to reveal inconsistencies between ranks that would explain the out-of-bounds memory access.
-   **Code Snippet:**
    ```python
    # In `forward` method
    valid_indices = torch.where(mask_flat)[0]

    # --- Begin Debug Logging ---
    rank = dist.get_rank() if dist.is_initialized() else 0
    log.info(f"[Rank {rank}] hidden_flat shape: {hidden_flat.shape}")
    log.info(f"[Rank {rank}] target_flat shape: {target_flat.shape}")
    # ... more logging ...
    # --- End Debug Logging ---

    if valid_indices.numel() == 0:
        continue

    valid_hidden = torch.index_select(hidden_flat, 0, valid_indices)
    valid_targets = torch.index_select(target_flat, 0, valid_indices)
    ```

## Attempt 7: Add Detailed Logging and Analyze

-   **Implementation**: To debug the recurring CUDA error, detailed, rank-specific logging was added to the `forward` method of the implementation from Attempt 5. This was done to inspect the shapes and value ranges of the tensors on each rank right before the failing `index_select` call.
-   **Result**: The logging was successful and revealed the root cause of the problem.

-   **Log Analysis:**
    ```
    INFO:torchtune.utils._logging:[Rank 5] hidden_flat shape: torch.Size([1024, 4096])
    INFO:torchtune.utils._logging:[Rank 5] target_flat shape: torch.Size([8192])
    ```
    The logs showed a critical shape mismatch on every rank. `hidden_flat` was correctly sized for a local batch (`1 * 1024`), but `target_flat` was sized for the global batch (`8 * 1024`). This caused `torch.index_select` to access out-of-bounds indices on `hidden_flat`.

## Proposed Solution: Robust Target Slicing

-   **Rationale**: The core issue is that the `targets` tensor is not being sharded before being passed to the loss function, unlike the model's `outputs`. The loss function can be made more robust by detecting this situation and manually slicing `targets` to get the correct local batch slice for the current rank.
-   **Implementation**:
    1.  In the `forward` method, compare the batch size of `targets` with the batch size of `local_outputs`.
    2.  If the `targets` batch size is larger, it's a global batch. Use the rank and local batch size to calculate the correct slice `targets[start:end]`.
    3.  Use this new `local_targets` tensor for all subsequent operations.
-   **Code Snippet:**
    ```python
    # In `forward` method
    if targets.shape[0] > local_outputs.shape[0]:
        local_bsz = local_outputs.shape[0]
        rank = dist.get_rank()
        start = rank * local_bsz
        end = start + local_bsz
        local_targets = targets[start:end]
    else:
        local_targets = targets

    target_chunks = local_targets.tensor_split(self.num_output_chunks, dim=1)
    ```

## Attempt 8: Robust Target Slicing

-   **Implementation**: Added logic to the `forward` method to detect when `targets` was a global tensor and slice it to get the appropriate local shard for the current rank before chunking and masking.
-   **Result**: The same `CUDA error: device-side assert triggered` occurred again, pointing to `torch.index_select`. This indicates that despite our efforts to align the batch dimensions, there is still a fundamental mismatch in how the indices are being generated or used across the different devices.

-   **Error Log:**
    ```
    /pytorch/aten/src/ATen/native/cuda/IndexKernelUtils.cu:16: vectorized_gather_kernel: ...
    Assertion `ind >=0 && ind < ind_dim_size && "vectorized gather kernel index out of bounds"` failed.
    ...
    torch.AcceleratorError: CUDA error: device-side assert triggered

    from user code:
    File "/data/users/felipemello/torchtune/torchtune/modules/loss/cross_entropy_loss.py", line 162, in forward
        valid_targets = torch.index_select(target_flat, 0, valid_indices)
    ```

-   **Code Snippet:**
    ```python
    # In `forward` method
    if targets.shape[0] > local_outputs.shape[0]:
        local_bsz = local_outputs.shape[0]
        rank = dist.get_rank()
        start = rank * local_bsz
        end = start + local_bsz
        local_targets = targets[start:end]
    else:
        local_targets = targets

    target_chunks = local_targets.tensor_split(self.num_output_chunks, dim=1)
    ```

## Attempt 9: Explicit Target Sharding for Tensor Parallelism

*   **Analysis**: The logs from the previous attempt clearly show a shape mismatch inside the `compute_cross_entropy` function, which is called for each chunk.
    ```
    INFO:torchtune.utils._logging:Hidden chunk shape: torch.Size([8, 1024, 4096])
    INFO:torchtune.utils._logging:Target chunk shape: torch.Size([8, 2048])
    ```
    The `hidden_chunk` comes from the model's output `DTensor`, which is correctly sharded along the sequence length dimension for tensor parallelism (8192 / 2 ranks = 4096 local sequence length, then chunked). However, the `targets` tensor is not sharded and remains at its global size (8192 sequence length) on each rank. When `targets` is chunked, the resulting `target_chunk` has a larger sequence length (2048) than the `hidden_chunk` (1024).

    This mismatch propagates down to the `torch.index_select` call inside the compiled `compute_cross_entropy` function, where indices derived from the oversized `target_flat` tensor are used to index the smaller `hidden_flat` tensor, causing an out-of-bounds memory access and the subsequent CUDA error.

*   **Rationale**: The fix is to make the loss function aware of this potential mismatch. Before chunking, we must ensure the `targets` tensor is correctly sharded to match the `outputs` tensor on each rank. By explicitly slicing the `targets` tensor based on the rank's portion of the data, we align the `hidden` and `target` chunks, resolving the indexing error.

*   **Proposed Solution**: In the `forward` method, right after converting the `outputs` DTensor to a local tensor, we will add logic to check if `targets` needs to be sharded. If the sequence length of `targets` is greater than the local sequence length of `outputs`, we will slice `targets` to get the correct local shard for the current rank.

*   **Code Snippet:**
    ```python
    # In LinearCrossEntropyLoss.forward()
    import torch.distributed as dist

    # ...
    is_dtensor_input = isinstance(outputs, DTensor)
    if is_dtensor_input:
        local_outputs = outputs.to_local()
        # If targets are not sharded but outputs are, shard targets manually.
        # This is a common case in TP where targets are broadcasted to all ranks.
        if targets.shape[1] > local_outputs.shape[1]:
            device_mesh = outputs.device_mesh
            rank = device_mesh.get_rank()

            # We assume TP shards along the sequence length dimension (dim 1)
            local_seq_len = local_outputs.shape[1]
            start_idx = rank * local_seq_len
            end_idx = start_idx + local_seq_len
            local_targets = targets[:, start_idx:end_idx].contiguous()
        else:
            local_targets = targets
    else:
        local_outputs = outputs
        local_targets = targets

    # --- Chunk along sequence dimension ---
    hidden_chunks = local_outputs.tensor_split(self.num_output_chunks, dim=1)
    target_chunks = local_targets.tensor_split(self.num_output_chunks, dim=1) # Use local_targets
    # ...
    ```

## Attempt 10: Padding to Fix Shape Inference Error

*   **Analysis**: The fix from Attempt 9 solved the target sharding issue but revealed a deeper problem within `torch.compile`. The new error is `Dynamo failed to run FX node with fake tensors: ... got RuntimeError('a and b must have same reduction dim, but got [8116, 8192] X [4096, 128256].')`.

    This error happens during the `aten.mm.default` (matrix multiply) operation inside the compiled `linear_projection` call. Critically, the logs show that while the *real* `valid_hidden` tensor has the correct shape (e.g., `[8116, 4096]` on one rank), the *fake* tensor used by TorchDynamo for shape propagation has an incorrect shape (`[8116, 8192]`). Dynamo appears to be erroneously doubling the embedding dimension (`4096 * 2 = 8192`).

    This happens because the number of valid tokens varies per rank (`8116` on one rank, `8124` on another), and passing these dynamically-shaped local tensors to a sharded module inside `torch.compile` triggers a shape inference bug.

*   **Rationale**: To work around the bug, we must ensure the tensors passed into the compiled region have the same shape on all ranks. The most direct way to achieve this is to communicate the number of valid tokens per chunk across ranks, find the maximum, and pad all tensors to that maximum size. The padded values in the `targets` tensor can be set to the `ignore_index` so they are correctly handled by the `cross_entropy` function.

*   **Proposed Solution**:
    1.  Refactor the loss class. The masking logic will be moved out of the compiled function and into the main `forward` method.
    2.  The compiled function, renamed to `_compute_proj_and_loss`, will only perform the projection and loss calculation.
    3.  In the `forward` method, after masking and creating `valid_hidden` for a chunk, we will use `dist.all_reduce` to find the maximum number of valid tokens across all TP ranks.
    4.  We will then pad `valid_hidden` (with zeros) and `valid_targets` (with `ignore_index`) to this maximum size.
    5.  Finally, we convert the padded `valid_hidden` into a replicated `DTensor` and pass it to the compiled `_compute_proj_and_loss` function. This provides uniform, consistent inputs to the compiled region, avoiding the shape inference error.

*   **Code Snippet:**
    ```python
    # In LinearCrossEntropyLoss
    def forward(self, outputs, targets):
        # ... setup and sharding logic ...
        for hidden_chunk, target_chunk in zip(hidden_chunks, target_chunks):
            # ... masking logic to get valid_hidden, valid_targets ...
            num_valid_this_chunk = torch.tensor(valid_indices.numel(), ...)

            if is_dtensor_input:
                # Get max valid tokens across ranks
                dist.all_reduce(num_valid_this_chunk, op=dist.ReduceOp.MAX, ...)
                max_valid = num_valid_this_chunk.item()

                # Pad to max size
                pad_size = max_valid - valid_hidden.shape[0]
                if pad_size > 0:
                    valid_hidden = F.pad(valid_hidden, (0, 0, 0, pad_size), "constant", 0)
                    valid_targets = F.pad(valid_targets, (0, pad_size), "constant", self.ignore_index)

                # Convert to replicated DTensor
                valid_hidden = DTensor.from_local(valid_hidden, mesh, [Replicate()])

            loss_sum_chunk = self._compute_proj_and_loss(valid_hidden, valid_targets)
            # ... accumulate loss ...

    def _compute_proj_and_loss(self, valid_hidden, valid_targets):
        # This part is compiled
        logits = self.linear_projection(valid_hidden)
        loss_sum = F.cross_entropy(logits.float(), valid_targets, reduction="sum", ...)
        return loss_sum
    ```

## Attempt 11: Remove Pre-Masking (User Suggestion)

*   **Analysis**: As an alternative to the complex padding solution, the user suggested removing the pre-masking logic entirely. Instead of selecting valid tokens before the linear projection, we would pass the full `hidden_chunk` to the projection layer and let the final `F.cross_entropy` call handle the `ignore_index`.

*   **Rationale**: This approach simplifies the code immensely. It removes the `torch.where` and `torch.index_select` calls, the cross-rank communication (`all_reduce`), and the padding logic. By feeding a consistently-shaped tensor (`hidden_chunk`) into the compiled region, it completely sidesteps the `torch.compile` shape inference bug.

*   **Trade-off**: The major drawback is performance and memory usage. The original goal of `LinearCrossEntropyLoss` was to avoid the expensive `[batch * seq_len, embed_dim] @ [embed_dim, vocab_size]` matrix multiplication for padding tokens. By removing the masking, we would be performing this large computation for all tokens, potentially leading to OOM errors and slower training, thus defeating the purpose of the module's memory-saving design.

*   **Proposed Solution**: Modify the `compute_cross_entropy` function to remove all masking logic. It would directly compute logits on the `hidden_chunk` and then calculate the loss.

*   **Code Snippet:**
    ```python
    # In LinearCrossEntropyLoss (if this approach is chosen)

    # The `compute_cross_entropy` method would be *compiled* and simplified:
    def compute_cross_entropy(self, hidden_chunk, target_chunk):
        hidden_flat = hidden_chunk.reshape(-1, hidden_chunk.size(-1))
        target_flat = target_chunk.reshape(-1)

        # No more masking here. Directly compute logits for all tokens.
        logits = self.linear_projection(hidden_flat)

        loss = F.cross_entropy(
            logits.float(),
            target_flat,
            reduction="sum",
            ignore_index=self.ignore_index,
        )
        # We would also need a way to count valid tokens for the final average.
        num_valid_tokens = (target_flat != self.ignore_index).sum()
        return loss, num_valid_tokens

    # The forward method would just loop and call the above.
    # No padding or cross-rank communication would be needed.
    ```

## Attempt 12: The Root Cause and Final No-Masking Fix

*   **Analysis**: The user tried the no-masking approach from Attempt 11 and received the same error as Attempt 10: `RuntimeError: a and b must have same reduction dim, but got [8192, 8192] X [4096, 128256].` This provides a critical insight: the error is not caused by dynamic shapes from masking. The root cause is a `torch.compile` shape inference bug that occurs whenever a **local tensor** is passed into a compiled region that contains a call to a **sharded module**. Dynamo incorrectly infers the shape of the local tensor, causing the `mm` to fail.

*   **Rationale**: The definitive solution, whether we use masking or not, is to avoid this buggy interaction. We must prepare all inputs for the compiled function in the eager-mode `forward` method. Specifically, any tensor that will interact with a sharded module must be converted to a `DTensor` *before* being passed to the compiled function. Creating a replicated `DTensor` is the safest option as it presents the same shape on all ranks.

*   **Proposed Solution (No-Masking)**:
    1.  Refactor the `forward` method. Inside the loop over chunks, it will:
        a.  Reshape the `hidden_chunk` into `hidden_flat`.
        b.  Explicitly convert the local `hidden_flat` tensor into a replicated `DTensor`.
    2.  Refactor the `compute_cross_entropy` method to accept the pre-flattened and replicated `DTensor` directly. This function will be compiled.
    3.  This approach is much simpler than the padding solution (Attempt 10) but still accepts the memory trade-off of not using pre-masking. It directly addresses the root cause of the compilation error.

*   **Code Snippet:**
    ```python
    # In LinearCrossEntropyLoss (final no-masking version)
    def forward(self, outputs, targets):
        # ... sharding logic ...
        total_loss = torch.tensor(0.0, ...)
        total_valid_tokens = torch.tensor(0, ...)
        mesh = outputs.device_mesh if is_dtensor_input else None

        for hidden_chunk, target_chunk in zip(hidden_chunks, target_chunks):
            hidden_flat = hidden_chunk.reshape(-1, hidden_chunk.size(-1))
            target_flat = target_chunk.reshape(-1)

            # Key Fix: Convert to DTensor *before* the compiled call
            if is_dtensor_input:
                hidden_flat = DTensor.from_local(hidden_flat, mesh, [Replicate()])

            loss_sum_chunk, num_valid_chunk = self.compute_cross_entropy(
                hidden_flat, target_flat
            )
            total_loss += loss_sum_chunk
            total_valid_tokens += num_valid_chunk
        # ... averaging ...

    # This gets compiled
    def compute_cross_entropy(self, hidden_flat, target_flat):
        # hidden_flat is now a DTensor, ready for the sharded module
        logits = self.linear_projection(hidden_flat)
        loss = F.cross_entropy(logits.float(), target_flat, ...)
        num_valid_tokens = (target_flat != self.ignore_index).sum()
        return loss, num_valid_tokens
    ```
