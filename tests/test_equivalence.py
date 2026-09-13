import copy
import torch

from tlora.multi_lora_linear import IndependentLoRALinear, MultiLoRALinear


def _copy_base_and_adapter(shared, independent, adapter_id):
    independent.base.weight.data.copy_(shared.base.weight.data)
    if shared.base.bias is not None:
        independent.base.bias.data.copy_(shared.base.bias.data)

    source = shared.adapters[str(adapter_id)]
    independent.adapter.A.data.copy_(source.A.data)
    independent.adapter.B.data.copy_(source.B.data)


def test_forward_equivalence():
    torch.manual_seed(7)

    ranks = {0: 2, 1: 4, 2: 8}
    shared = MultiLoRALinear(16, 12, ranks)

    refs = {
        aid: IndependentLoRALinear(16, 12, rank)
        for aid, rank in ranks.items()
    }
    for aid, ref in refs.items():
        _copy_base_and_adapter(shared, ref, aid)

    # Give B non-zero values so we test real LoRA deltas, not only the zero-init case.
    for adapter in shared.adapters.values():
        torch.nn.init.normal_(adapter.B, mean=0.0, std=0.02)
    for aid, ref in refs.items():
        ref.adapter.B.data.copy_(shared.adapters[str(aid)].B.data)

    x = torch.randn(9, 16)
    adapter_ids = torch.tensor([0, 1, 2, 0, 2, 1, 0, 1, 2])

    y_shared = shared(x, adapter_ids)

    y_ref = torch.empty_like(y_shared)
    for aid, ref in refs.items():
        mask = adapter_ids == aid
        y_ref[mask] = ref(x[mask])

    max_error = (y_shared - y_ref).abs().max().item()
    print(f"forward max abs error: {max_error:.3e}")
    assert torch.allclose(y_shared, y_ref, atol=1e-6, rtol=1e-6)


def test_gradient_equivalence():
    torch.manual_seed(11)

    ranks = {0: 2, 1: 4, 2: 8}
    shared = MultiLoRALinear(10, 7, ranks)

    refs = {
        aid: IndependentLoRALinear(10, 7, rank)
        for aid, rank in ranks.items()
    }
    for aid, ref in refs.items():
        _copy_base_and_adapter(shared, ref, aid)

    for adapter in shared.adapters.values():
        torch.nn.init.normal_(adapter.B, mean=0.0, std=0.02)
    for aid, ref in refs.items():
        ref.adapter.B.data.copy_(shared.adapters[str(aid)].B.data)

    x = torch.randn(12, 10)
    adapter_ids = torch.tensor([0, 0, 1, 2, 1, 0, 2, 2, 1, 0, 1, 2])
    target = torch.randn(12, 7)

    # Important: use SUM-reduction so per-job independent losses exactly equal
    # the aggregate shared loss.
    y = shared(x, adapter_ids)
    loss = torch.nn.functional.mse_loss(y, target, reduction="sum")
    loss.backward()

    for aid, ref in refs.items():
        mask = adapter_ids == aid
        y_ref = ref(x[mask])
        ref_loss = torch.nn.functional.mse_loss(
            y_ref, target[mask], reduction="sum"
        )
        ref_loss.backward()

        shared_adapter = shared.adapters[str(aid)]

        a_err = (shared_adapter.A.grad - ref.adapter.A.grad).abs().max().item()
        b_err = (shared_adapter.B.grad - ref.adapter.B.grad).abs().max().item()

        print(f"adapter {aid}: A grad error={a_err:.3e}, B grad error={b_err:.3e}")

        assert torch.allclose(
            shared_adapter.A.grad, ref.adapter.A.grad, atol=1e-6, rtol=1e-6
        )
        assert torch.allclose(
            shared_adapter.B.grad, ref.adapter.B.grad, atol=1e-6, rtol=1e-6
        )


def test_optimizer_isolation():
    torch.manual_seed(13)

    shared = MultiLoRALinear(8, 6, {0: 2, 1: 4, 2: 8})

    # Make all adapters active/non-zero.
    for adapter in shared.adapters.values():
        torch.nn.init.normal_(adapter.B, mean=0.0, std=0.02)

    before_0 = copy.deepcopy(shared.adapters["0"].state_dict())
    before_1 = copy.deepcopy(shared.adapters["1"].state_dict())
    before_2 = copy.deepcopy(shared.adapters["2"].state_dict())

    # Job 1 has its OWN optimiser, as required by the Shared Super-Model idea.
    optimizer_1 = torch.optim.SGD(shared.adapter_parameters(1), lr=0.1)

    x = torch.randn(5, 8)
    ids = torch.ones(5, dtype=torch.long)
    target = torch.randn(5, 6)

    optimizer_1.zero_grad()
    loss = torch.nn.functional.mse_loss(shared(x, ids), target)
    loss.backward()
    optimizer_1.step()

    after_0 = shared.adapters["0"].state_dict()
    after_1 = shared.adapters["1"].state_dict()
    after_2 = shared.adapters["2"].state_dict()

    # Adapter 1 must change.
    changed_1 = any(
        not torch.equal(before_1[k], after_1[k]) for k in before_1
    )
    assert changed_1

    # Adapters 0 and 2 must remain bit-for-bit unchanged.
    for k in before_0:
        assert torch.equal(before_0[k], after_0[k])
    for k in before_2:
        assert torch.equal(before_2[k], after_2[k])

    print("optimizer isolation: PASS")


if __name__ == "__main__":
    test_forward_equivalence()
    test_gradient_equivalence()
    test_optimizer_isolation()
    print("\nALL PHASE-1 TESTS PASSED")
