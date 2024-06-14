import logging
from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from mmsd.dataset import MMSDModelInput
from mmsd.model import MMSDOutput

logger = logging.getLogger("mmsd.predictor")


class MemoEnhancedPredictor(nn.Module):

    def __init__(
        self,
        model: Callable[..., MMSDOutput],
        use_memo: bool = True,
        memo_size: int = 512,
        embed_size: int = 512,
    ) -> None:
        super().__init__()
        self.model = model
        self.use_memo = use_memo
        if use_memo:
            self.memo_size = memo_size
            self.register_buffer(
                "entropy", torch.zeros(2, memo_size, device=model.device)
            )
            self.register_buffer(
                "embed_memo",
                torch.zeros(2, memo_size, embed_size, device=model.device),
            )
            self.register_buffer(
                "memo_ptr", torch.zeros(2, dtype=torch.long, device=model.device)
            )

    def write(
        self,
        model_out: MMSDOutput,
        pseudo_y: Tensor,
        entropy: Tensor,
        label: int,
    ) -> None:
        memo_ptr: int = self.memo_ptr[label].item()

        selected_size = (pseudo_y == label).sum().item()
        selected_embeds = model_out.fused_embeds[pseudo_y == label]
        selected_entropy = entropy[pseudo_y == label]

        if memo_ptr == self.memo_size:
            memo_sorted_entropy, memo_sorted_idx = torch.sort(
                self.entropy[label], descending=True
            )
            sorted_entropy, sorted_idx = torch.sort(selected_entropy)

            if selected_size > self.memo_size:
                sorted_entropy = sorted_entropy[: self.memo_size]
                sorted_idx = sorted_idx[: self.memo_size]
            else:
                memo_sorted_entropy = memo_sorted_entropy[:selected_size]
                memo_sorted_idx = memo_sorted_idx[:selected_size]

            ioi = memo_sorted_entropy > sorted_entropy
            if ioi.size(0) == 0:
                return

            sorted_idx = sorted_idx[ioi]
            need_to_replace_idx = memo_sorted_idx[ioi]

            self.embed_memo[label, need_to_replace_idx] = selected_embeds[sorted_idx]
            self.entropy[label, need_to_replace_idx] = selected_entropy[sorted_idx]
        else:
            end_idx = memo_ptr + selected_size
            if end_idx > self.memo_size:
                need_size = self.memo_size - memo_ptr
                sorted_entropy, sorted_idx = torch.sort(selected_entropy)
                sorted_idx = sorted_idx[:need_size]
                selected_entropy = sorted_entropy[:need_size]
                selected_embeds = selected_embeds[sorted_idx]
                end_idx = self.memo_size

            self.embed_memo[label, memo_ptr:end_idx] = selected_embeds
            self.entropy[label, memo_ptr:end_idx] = selected_entropy
            self.memo_ptr[label] = end_idx

    def forward(self, batch: MMSDModelInput) -> tuple[Tensor, Tensor, Tensor]:
        output = self.model(**batch, return_loss=False)

        logits = output.logits
        pred = F.softmax(logits, dim=-1)

        if not self.use_memo:
            return pred

        log_pred = F.log_softmax(logits, dim=-1)

        entropy = -torch.sum(pred * log_pred, dim=-1)
        pseudo_y = torch.argmax(pred, dim=-1)

        self.write(output, pseudo_y, entropy, 0)
        self.write(output, pseudo_y, entropy, 1)

        cosin = torch.einsum("bd,cmd->bmc", output.fused_embeds, self.embed_memo)
        cosin = cosin.sum(dim=1)
        memo_pred = F.softmax(cosin, dim=-1)
        return memo_pred, pred, entropy
