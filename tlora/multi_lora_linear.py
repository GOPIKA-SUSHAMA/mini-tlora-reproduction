import math
from typing import Dict, Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F


class LoRAAdapter(nn.Module):
    """A minimal LoRA branch: delta(x) = scaling * (x @ A^T) @ B^T."""

    def __init__(self, in_features: int, out_features: int, rank: int, alpha: float | None = None):
        super().__init__()
        if rank <= 0:
            raise ValueError("rank must be > 0")

        self.rank = rank
        self.alpha = float(alpha if alpha is not None else rank)
        self.scaling = self.alpha / self.rank

        # A: [rank, in_features], B: [out_features, rank]
        self.A = nn.Parameter(torch.empty(rank, in_features))
        self.B = nn.Parameter(torch.zeros(out_features, rank))

        # Standard LoRA-style init: A random, B zero -> initial delta is exactly zero.
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = F.linear(x, self.A)          # [..., rank]
        delta = F.linear(hidden, self.B)      # [..., out_features]
        return delta * self.scaling


class IndependentLoRALinear(nn.Module):
    """
    Reference implementation for ONE LoRA job.
    The backbone parameters are frozen; only A/B train.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int,
        bias: bool = True,
        alpha: float | None = None,
    ):
        super().__init__()
        self.base = nn.Linear(in_features, out_features, bias=bias)
        for p in self.base.parameters():
            p.requires_grad_(False)

        self.adapter = LoRAAdapter(in_features, out_features, rank, alpha)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(x) + self.adapter(x)


class MultiLoRALinear(nn.Module):
    """
    Phase-1 Shared Super-Model primitive.

    One frozen backbone is shared by all jobs. Each sample is routed to one
    independently trainable adapter. Adapters may have heterogeneous ranks.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        adapter_ranks: Dict[int, int],
        bias: bool = True,
    ):
        super().__init__()
        if not adapter_ranks:
            raise ValueError("adapter_ranks cannot be empty")

        self.base = nn.Linear(in_features, out_features, bias=bias)
        for p in self.base.parameters():
            p.requires_grad_(False)

        self.adapters = nn.ModuleDict(
            {
                str(adapter_id): LoRAAdapter(in_features, out_features, rank)
                for adapter_id, rank in adapter_ranks.items()
            }
        )

    def adapter_parameters(self, adapter_id: int) -> Iterable[nn.Parameter]:
        return self.adapters[str(adapter_id)].parameters()

    def forward(self, x: torch.Tensor, adapter_ids: torch.Tensor) -> torch.Tensor:
        """
        x:
          [batch, in_features]

        adapter_ids:
          [batch], integer adapter/job ID for each sample.
        """
        if x.ndim != 2:
            raise ValueError("Phase 1 expects x shaped [batch, in_features]")
        if adapter_ids.ndim != 1 or adapter_ids.shape[0] != x.shape[0]:
            raise ValueError("adapter_ids must be [batch] and match x.shape[0]")

        # Shared frozen-backbone computation.
        out = self.base(x)

        # Correctness-first implementation.
        # Phase 2 will replace this Python loop with a heterogeneous fused kernel.
        result = out.clone()
        for adapter_id_str, adapter in self.adapters.items():
            adapter_id = int(adapter_id_str)
            mask = adapter_ids == adapter_id

            if bool(mask.any()):
                result[mask] = result[mask] + adapter(x[mask])

        return result
