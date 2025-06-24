# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn
from torch.distributed.tensor import DTensor, Replicate

from torchtune.modules.loss.loss_types import SFTLoss
from torchtune.utils import get_logger

log = get_logger()


# DO NOT CHANGE THIS CLASS. IF IMPLEMENTING A NEW LOGIC, CREATE A NEW CLASS.
class NEWLinearCrossEntropyLoss(nn.Module, SFTLoss):
    """
    Cross-entropy loss that combines the linear projection with the loss calculation.
    It computes logits for all tokens and uses the `ignore_index` in `F.cross_entropy`
    to mask out labels.

    This version is simpler and avoids torch.compile issues with dynamic shapes
    by passing a replicated DTensor into the compiled region, but is less
    memory-efficient than pre-masking.

    You need to skip the final projection layer in your model and pass it to the loss instead.
    """

    def __init__(
        self,
        num_output_chunks: int = 4,
        ignore_index: int = -100,
    ):
        super().__init__()
        self.linear_projection = None
        self.num_output_chunks = num_output_chunks
        self.ignore_index = ignore_index

    def apply_compile_strategy(self, *args, **kwargs):
        """Applies torch.compile to the core computation."""
        self.compute_cross_entropy = torch.compile(
            self.compute_cross_entropy, *args, **kwargs
        )
        return self

    def set_model_output(self, model: nn.Module) -> None:
        """Modify model output to match the expected input for the loss function."""
        model.skip_output_layer = True
        self.linear_projection = model.output
        log.info(
            f"Linear projection weight shape: {self.linear_projection.weight.shape}"
        )

    def compute_cross_entropy(
        self,
        hidden_flat: torch.Tensor,
        target_flat: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Computes logits for the entire chunk and then cross-entropy loss.
        The `ignore_index` is handled by `F.cross_entropy`.

        Args:
            hidden_flat (torch.Tensor): Flattened hidden state from the model.
                Can be a replicated ``DTensor``. Shape ``[batch * chunk_size, embed_dim]``
            target_flat (torch.Tensor): Flattened labels for the model.
                Shape ``[batch * chunk_size]``

        Returns:
            A tuple of (sum of loss, number of valid tokens).
        """
        if self.linear_projection is None:
            raise AttributeError(
                "Loss function was not properly configured. "
                "Ensure set_model_output() is called before the forward pass."
            )

        # [batch * chunk_size, embed_dim] @ [embed_dim, vocab_size]
        logits = self.linear_projection(hidden_flat)

        loss_sum = F.cross_entropy(
            logits.float(),
            target_flat,
            reduction="sum",
            ignore_index=self.ignore_index,
        )

        # Manually count valid tokens for the final average
        num_valid_tokens = (target_flat != self.ignore_index).sum()

        return loss_sum, num_valid_tokens

    def forward(
        self,
        outputs: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """
        The forward method handles tensor sharding and chunking before calling
        the compiled `compute_cross_entropy` function.

        Args:
            outputs (torch.Tensor): Hidden state from the model. Shape [bsz, seq_len, emb_dim]
            targets (torch.Tensor): Labels. Shape [bsz, seq_len]

        Returns:
            The final, averaged loss.
        """
        # --- DTensor Sharding ---
        is_dtensor_input = isinstance(outputs, DTensor)
        if is_dtensor_input:
            local_outputs = outputs.to_local()
            mesh = outputs.device_mesh
            
            # Check if targets need to be sharded to match local outputs
            # In TP, targets are typically replicated but we need to slice them
            # to match the local sequence length
            if targets.shape[1] > local_outputs.shape[1]:
                tp_dim = mesh.mesh_dim_names.index("tp")
                tp_rank = mesh.get_coordinate()[tp_dim]

                local_seq_len = local_outputs.shape[1]
                start_idx = tp_rank * local_seq_len
                end_idx = start_idx + local_seq_len
                local_targets = targets[:, start_idx:end_idx].contiguous()
            else:
                local_targets = targets
        else:
            local_outputs = outputs
            local_targets = targets

        # --- Chunking & Loss Computation ---
        hidden_chunks = local_outputs.tensor_split(self.num_output_chunks, dim=1)
        target_chunks = local_targets.tensor_split(self.num_output_chunks, dim=1)

        total_loss = torch.tensor(0.0, device=local_outputs.device)
        total_valid_tokens = torch.tensor(0, device=local_outputs.device)

        for hidden_chunk, target_chunk in zip(hidden_chunks, target_chunks):
            hidden_flat = hidden_chunk.reshape(-1, hidden_chunk.size(-1))
            target_flat = target_chunk.reshape(-1)

            # Key Fix: Convert to DTensor *before* the compiled call
            if is_dtensor_input:
                hidden_flat = DTensor.from_local(hidden_flat, mesh, [Replicate()])

            loss_sum, num_valid_tokens = self.compute_cross_entropy(
                hidden_flat, target_flat
            )
            total_loss += loss_sum
            total_valid_tokens += num_valid_tokens

        # --- Final Averaging ---
        # If distributed, we must sum the local losses and token counts across all ranks.
        if is_dtensor_input:
            # Get the tensor parallel process group from the device mesh
            # to ensure we only reduce across TP ranks. The recipe will handle DP reduction.
            tp_group = mesh.get_group(mesh_dim="tp")
            
            # Fix: Detach tensors before all_reduce to avoid autograd issues
            # All_reduce operations should not participate in gradient computation
            loss_for_reduce = total_loss.detach().clone()
            tokens_for_reduce = total_valid_tokens.detach().clone()
            
            dist.all_reduce(loss_for_reduce, op=dist.ReduceOp.SUM, group=tp_group)
            dist.all_reduce(tokens_for_reduce, op=dist.ReduceOp.SUM, group=tp_group)
            
            # Use the reduced values for final computation
            final_loss = torch.where(
                tokens_for_reduce > 0,
                loss_for_reduce / tokens_for_reduce,
                torch.tensor(0.0, device=total_loss.device),
            )
        else:
            final_loss = torch.where(
                total_valid_tokens > 0,
                total_loss / total_valid_tokens,
                torch.tensor(0.0, device=total_loss.device),
            )

        return final_loss



# old loss


class OLDLinearCrossEntropyLoss(nn.Module, SFTLoss):
    def __init__(
        self,
        num_output_chunks: int = 8,
        ignore_index: int = -100,
    ):
        super().__init__()
        self.linear_projection = None
        self.num_output_chunks = num_output_chunks
        self.ignore_index = ignore_index

    def apply_compile_strategy(self, *args, **kwargs):
        log.warning("Skipping compile loss, as it is not supported at this time")
        # TODO fix compile and re-enable
        # self.compute_cross_entropy = torch.compile(
        #     self.compute_cross_entropy, *args, **kwargs
        # )
        return self

    def set_model_output(self, model: nn.Module) -> None:
        model.skip_output_layer = True
        self.linear_projection = model.output

    def compute_cross_entropy(
        self,
        hidden_chunk: torch.Tensor,
        target_chunk: torch.Tensor,
    ) -> torch.Tensor:
        # Select hidden states and targets where mask is True
        mask_chunk = target_chunk != self.ignore_index
        if mask_chunk.sum() == 0:
            # Unmask 1 token to allow loss to sync with all data parallel workers
            mask_chunk[0] = True

        target_chunk = target_chunk[mask_chunk]  # [num_valid]
        if isinstance(hidden_chunk, DTensor):
            # DTensor doesn't support masks so we have to mask locally
            mesh = hidden_chunk.device_mesh
            placements = hidden_chunk.placements
            local_hidden_chunk = hidden_chunk.to_local()[mask_chunk]
            hidden_chunk = DTensor.from_local(
                local_hidden_chunk, mesh, placements
            )  # [num_valid, embed_dim]
        else:
            hidden_chunk = hidden_chunk[mask_chunk]  # [num_valid, embed_dim]

        # [num_valid, embed_dim] @ [embed_dim, vocab_size]
        if self.linear_projection is None:
            raise AttributeError("forward called before update_model")
        logits = self.linear_projection(hidden_chunk)  # [num_valid, vocab_size]
        if isinstance(logits, DTensor):
            logits = logits.full_tensor()

        return F.cross_entropy(
            logits.float(),
            target_chunk,
            reduction="sum",
            ignore_index=self.ignore_index,
        )

    def forward(
        self,
        outputs: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        # Total number of non-ignored tokens across the entire batch
        mask = targets != self.ignore_index
        total_elements = mask.sum()

        # Chunk along sequence dimension
        hidden_chunks = outputs.tensor_split(self.num_output_chunks, dim=1)
        target_chunks = targets.tensor_split(self.num_output_chunks, dim=1)

        # Compute cross-entropy loss for the chunks
        total_loss = 0.0
        for idx in range(len(hidden_chunks)):
            total_loss += self.compute_cross_entropy(
                hidden_chunks[idx],
                target_chunks[idx],
            )

        if total_elements == 0:
            # must return after calling compute_cross_entropy to not hang during data parallel training
            return total_loss
        else:
            return total_loss / total_elements