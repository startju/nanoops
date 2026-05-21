"""CPU offload of optimizer state for nanchat's DistMuonAdamW.

Why this exists: d24+B=1 on 2× RTX 3090 hits a reproducible 1.35 GiB
allocator-fragmentation OOM at iter 3 (with naive SDPA) / iter 22
(with L-attn checkpoint). All Python-level activation tricks (chunked
SDPA, MLP/attn checkpoint, LSE-only ctx) failed to escape — the peak
transient during SDPA backward is fundamentally bounded below.

The remaining lever is the persistent optim state pool. nanchat already
ZeRO-1 shards state across ranks (each rank holds 1/world_size of large-
param state), so the GPU footprint per rank is ~half of the total. CPU
offload moves that remaining slice to host pinned memory; per optimizer
step we async-copy state to GPU buffers, run the existing fused kernel,
and async-copy results back.

Cost: ~PCIe round-trip of the per-rank state slice. For nanchat d24
(~2.5 GB Muon state + ~300 MB AdamW state per rank under ZeRO-1) on
PCIe 4.0 x16 (~25 GB/s peak, ~12 GB/s sustained over both copies and
allocator overhead), one optimizer step adds roughly 200-400 ms — once
per training iter (after 256 accum), so ~+0.5% wall time on a 65 s/iter
d24 schedule.

Benefit: ~2.8 GB of GPU memory freed per rank, well above the ~1.5 GB
margin we need to clear the d24 OOM cliff.

Wiring: when NANOOPS_OFFLOAD_OPTIM=1 is set, `nanoops.integration._apply`
monkey-patches DistMuonAdamW._compute_adamw and ._compute_muon with the
functions defined here. The patch is reversed by _restore via the
normal originals dict.

Limitations:
  - Only handles DistMuonAdamW (the distributed path nanchat uses on >1 GPU).
    Single-GPU MuonAdamW is left alone — single-GPU training doesn't have
    the ZeRO halving and OOM behavior is different there anyway.
  - State loaded from checkpoint comes back on whichever device the
    checkpoint stored. We don't try to re-pin it; first optim step after
    resume re-pays the H2D + D2H cost from a non-pinned CPU buffer.
"""

from __future__ import annotations

import torch


def patched_compute_adamw(self, group, info, gather_list, rank, world_size):
    """CPU-offloaded version of DistMuonAdamW._compute_adamw.

    State (exp_avg, exp_avg_sq) lives in pinned CPU memory; per optimizer
    step we H2D copy to GPU buffers, run the fused AdamW kernel, then
    D2H back.
    """
    import torch.distributed as dist
    from nanochat.optim import adamw_step_fused

    param_infos = info['param_infos']
    for p in group['params']:
        pinfo = param_infos[p]
        pinfo['future'].wait()
        grad_slice = pinfo['grad_slice']
        state = self.state[p]

        if pinfo['is_small']:
            p_slice = p
        else:
            rank_size = p.shape[0] // world_size
            p_slice = p[rank * rank_size:(rank + 1) * rank_size]

        if not state:
            state['step'] = 0
            # Allocate on CPU pinned memory — pin_memory enables async H2D/D2H
            state['exp_avg'] = torch.zeros(
                p_slice.shape, dtype=p_slice.dtype, device="cpu", pin_memory=True,
            )
            state['exp_avg_sq'] = torch.zeros(
                p_slice.shape, dtype=p_slice.dtype, device="cpu", pin_memory=True,
            )
        state['step'] += 1

        # H2D — async because state is pinned
        exp_avg_g = state['exp_avg'].to(p.device, non_blocking=True)
        exp_avg_sq_g = state['exp_avg_sq'].to(p.device, non_blocking=True)

        self._adamw_step_t.fill_(state['step'])
        self._adamw_lr_t.fill_(group['lr'])
        self._adamw_beta1_t.fill_(group['betas'][0])
        self._adamw_beta2_t.fill_(group['betas'][1])
        self._adamw_eps_t.fill_(group['eps'])
        self._adamw_wd_t.fill_(group['weight_decay'])

        adamw_step_fused(
            p_slice, grad_slice, exp_avg_g, exp_avg_sq_g,
            self._adamw_step_t, self._adamw_lr_t, self._adamw_beta1_t,
            self._adamw_beta2_t, self._adamw_eps_t, self._adamw_wd_t,
        )

        # D2H — async copy back to pinned CPU buffer
        state['exp_avg'].copy_(exp_avg_g, non_blocking=True)
        state['exp_avg_sq'].copy_(exp_avg_sq_g, non_blocking=True)

        if not pinfo['is_small']:
            future = dist.all_gather_into_tensor(p, p_slice, async_op=True).get_future()
            gather_list.append(dict(future=future, params=None))


def patched_compute_muon(self, group, info, gather_list, rank):
    """CPU-offloaded version of DistMuonAdamW._compute_muon.

    Muon state buffers (momentum_buffer, second_momentum_buffer) live on
    CPU pinned memory; only the OWNED slice is copied H2D per step
    (non-owned chunks don't participate on this rank).
    """
    import torch.distributed as dist
    from nanochat.optim import muon_step_fused

    info['future'].wait()
    params = group['params']
    chunk_size = info['chunk_size']
    grad_chunk = info['grad_chunk']
    p = params[0]
    shape, device, dtype = p.shape, p.device, p.dtype

    start_idx = rank * chunk_size
    num_owned = min(chunk_size, max(0, len(params) - start_idx))

    state = self.state[p]
    if "momentum_buffer" not in state:
        state["momentum_buffer"] = torch.zeros(
            chunk_size, *shape, dtype=dtype, device="cpu", pin_memory=True,
        )
    if "second_momentum_buffer" not in state:
        state_shape = (
            (chunk_size, shape[-2], 1)
            if shape[-2] >= shape[-1]
            else (chunk_size, 1, shape[-1])
        )
        state["second_momentum_buffer"] = torch.zeros(
            state_shape, dtype=dtype, device="cpu", pin_memory=True,
        )
    red_dim = -1 if shape[-2] >= shape[-1] else -2

    updated_params = torch.empty(chunk_size, *shape, dtype=dtype, device=device)

    if num_owned > 0:
        # H2D the OWNED slice only — saves PCIe bandwidth for non-owned chunks
        mom_g = state["momentum_buffer"][:num_owned].to(device, non_blocking=True)
        mom2_g = state["second_momentum_buffer"][:num_owned].to(device, non_blocking=True)

        owned_params = [params[start_idx + i] for i in range(num_owned)]
        stacked_owned = torch.stack(owned_params)

        self._muon_momentum_t.fill_(group["momentum"])
        self._muon_beta2_t.fill_(group["beta2"])
        self._muon_lr_t.fill_(group["lr"] * max(1.0, shape[-2] / shape[-1]) ** 0.5)
        self._muon_wd_t.fill_(group["weight_decay"])

        muon_step_fused(
            grad_chunk[:num_owned], stacked_owned,
            mom_g, mom2_g,
            self._muon_momentum_t, self._muon_lr_t, self._muon_wd_t, self._muon_beta2_t,
            group["ns_steps"], red_dim,
        )
        updated_params[:num_owned].copy_(stacked_owned)

        # D2H — async copy back
        state["momentum_buffer"][:num_owned].copy_(mom_g, non_blocking=True)
        state["second_momentum_buffer"][:num_owned].copy_(mom2_g, non_blocking=True)

    if num_owned < chunk_size:
        updated_params[num_owned:].zero_()

    stacked_params = info["stacked_grads"]
    future = dist.all_gather_into_tensor(
        stacked_params, updated_params, async_op=True
    ).get_future()
    gather_list.append(dict(future=future, stacked_params=stacked_params, params=params))
