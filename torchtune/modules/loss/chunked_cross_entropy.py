# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch.nn as nn


class ChunkedCrossEntropyLoss(nn.Module):
    def __init__(self, chunk_size=1024, ignore_index=-100):
        super(ChunkedCrossEntropyLoss, self).__init__()
        self.chunk_size = chunk_size
        self.ignore_index = ignore_index
        self.cross_entropy = nn.CrossEntropyLoss(
            ignore_index=ignore_index, reduction="none"
        )

    def forward(self, logits, labels):
        # Split logits and labels into chunks along the last dimension (vocab_size)
        total_elements = (labels != self.ignore_index).sum()
        logits = logits.split(self.chunk_size, dim=-1)
        labels = labels.split(self.chunk_size, dim=-1)
        total_loss = 0.0

        for logits_chunk, labels_chunk in zip(logits, labels):
            # Compute the loss for the current chunk
            loss_chunk = self.cross_entropy(logits_chunk, labels_chunk)
            total_loss += loss_chunk.sum()

        return total_loss / total_elements
