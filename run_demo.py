import torch
from tlora.multi_lora_linear import MultiLoRALinear


torch.manual_seed(42)

model = MultiLoRALinear(
    in_features=16,
    out_features=8,
    adapter_ranks={
        0: 2,
        1: 4,
        2: 8,
        3: 16,
    },
)

print("Frozen backbone:")
for name, p in model.base.named_parameters():
    print(f"  {name:10s} requires_grad={p.requires_grad}")

print("\nIndependent heterogeneous adapters:")
for adapter_id, adapter in model.adapters.items():
    n = sum(p.numel() for p in adapter.parameters())
    print(f"  job={adapter_id} rank={adapter.rank:2d} trainable_params={n}")

x = torch.randn(8, 16)
adapter_ids = torch.tensor([0, 1, 2, 3, 0, 1, 2, 3])

y = model(x, adapter_ids)

print("\nInput :", tuple(x.shape))
print("Output:", tuple(y.shape))
print("Routes:", adapter_ids.tolist())
print("\nPhase-1 Shared Super-Model forward pass: PASS")
