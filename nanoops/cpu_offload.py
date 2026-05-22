"""CPU offload of optimizer state for nanchat's DistMuonAdamW.

Drop-in replacements for `DistMuonAdamW._compute_adamw` and
`._compute_muon` that allocate optim state in pinned CPU memory and
async-copy H2D / D2H around the existing fused kernels.

Wired in by `nanoops.integration._apply` when `NANOOPS_OFFLOAD_OPTIM=1`;
reversed by `_restore` via the standard `originals` dict.

Both classes are handled:
  - `DistMuonAdamW` (distributed, >1 GPU): patches `_compute_adamw` and
    `_compute_muon` — state lives sharded across ranks via ZeRO-1, this
    moves each rank's slice to CPU.
  - `MuonAdamW` (single GPU): patches `_step_adamw` and `_step_muon` —
    full-size state lives on CPU (no ZeRO sharding to begin with), the
    only path to fit a model that otherwise overflows VRAM.

Limitation: state loaded from a checkpoint comes back on whichever
device the checkpoint stored. The first optim step after resume re-pays
the H2D + D2H cost from a non-pinned CPU buffer; subsequent steps
re-pin via the lazy alloc path.
"""

from __future__ import annotations

import torch


def patched_compute_adamw(self, group, info, gather_list, rank, world_size):
    """CPU-offloaded version of DistMuonAdamW._compute_adamw.

    State (exp_avg, exp_avg_sq) lives in pinned CPU memory; per optimizer
    step we H2D copy to GPU buffers, run the fused AdamW kernel, then
    D2H back. empty_cache at entry releases the 256-fwd+bwd allocator
    cache so the per-param H2D copies can find contiguous space; same
    rationale as the MuonAdamW versions below. NOTE: torchrun always
    sets RANK/LOCAL_RANK/WORLD_SIZE in env, so NPROC=1 *also* uses
    DistMuonAdamW (not single-GPU MuonAdamW) — meaning this is the
    code path that runs in BOTH dual-GPU and single-GPU cases.
    """
    import torch.distributed as dist
    from nanochat.optim import adamw_step_fused

    torch.cuda.empty_cache()

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
        del exp_avg_g, exp_avg_sq_g

        if not pinfo['is_small']:
            future = dist.all_gather_into_tensor(p, p_slice, async_op=True).get_future()
            gather_list.append(dict(future=future, params=None))

    # Release per-param H2D transients back to driver so the next 256
    # fwd+bwd cycles get a clean allocator. CRITICAL: must synchronize
    # FIRST — the per-param D2H copies were issued with non_blocking=True
    # so the GPU buffers are still bound to pending stream events. The
    # CUDA allocator refuses to release blocks with outstanding events,
    # so empty_cache without sync is a no-op for those transients.
    torch.cuda.synchronize()
    torch.cuda.empty_cache()


def patched_compute_muon(self, group, info, gather_list, rank):
    """CPU-offloaded version of DistMuonAdamW._compute_muon.

    Muon state buffers (momentum_buffer, second_momentum_buffer) live on
    CPU pinned memory; only the OWNED slice is copied H2D per step
    (non-owned chunks don't participate on this rank). NOTE: when
    world_size=1 (NPROC=1 single-GPU runs via torchrun), this code
    H2D's the FULL state slice — ~5 GiB on d24, which is why we need
    the empty_cache at end to release the transient back to the driver.
    """
    import torch.distributed as dist
    from nanochat.optim import muon_step_fused

    torch.cuda.empty_cache()

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
        del mom_g, mom2_g, stacked_owned

    if num_owned < chunk_size:
        updated_params[num_owned:].zero_()

    stacked_params = info["stacked_grads"]
    future = dist.all_gather_into_tensor(
        stacked_params, updated_params, async_op=True
    ).get_future()
    gather_list.append(dict(future=future, stacked_params=stacked_params, params=params))

    # Release per-step transient (mom_g, mom2_g, stacked_owned, updated_params)
    # back to the driver. Same sync-before-empty_cache rationale as in
    # patched_compute_adamw: updated_params is referenced by the still-
    # pending async all_gather_into_tensor, and mom_g/mom2_g's storages
    # are bound to async D2H stream events. Without sync, empty_cache
    # can't release any of them.
    del updated_params
    torch.cuda.synchronize()
    torch.cuda.empty_cache()


# -----------------------------------------------------------------------------
# Single-GPU (MuonAdamW) versions: full-size state on CPU pinned memory.
# Same H2D/D2H pattern but without the reduce_scatter / all_gather glue.

def patched_step_adamw(self, group):
    """CPU-offloaded version of MuonAdamW._step_adamw (single-GPU path)."""
    from nanochat.optim import adamw_step_fused

    # empty_cache at start: returns the 256-fwd+bwd allocator cache to
    # the driver so the per-param H2D copies can find contiguous space.
    torch.cuda.empty_cache()

    for p in group['params']:
        if p.grad is None:
            continue
        grad = p.grad
        state = self.state[p]

        if not state:
            state['step'] = 0
            state['exp_avg'] = torch.zeros(
                p.shape, dtype=p.dtype, device="cpu", pin_memory=True,
            )
            state['exp_avg_sq'] = torch.zeros(
                p.shape, dtype=p.dtype, device="cpu", pin_memory=True,
            )
        state['step'] += 1

        exp_avg_g = state['exp_avg'].to(p.device, non_blocking=True)
        exp_avg_sq_g = state['exp_avg_sq'].to(p.device, non_blocking=True)

        self._adamw_step_t.fill_(state['step'])
        self._adamw_lr_t.fill_(group['lr'])
        self._adamw_beta1_t.fill_(group['betas'][0])
        self._adamw_beta2_t.fill_(group['betas'][1])
        self._adamw_eps_t.fill_(group['eps'])
        self._adamw_wd_t.fill_(group['weight_decay'])

        adamw_step_fused(
            p, grad, exp_avg_g, exp_avg_sq_g,
            self._adamw_step_t, self._adamw_lr_t, self._adamw_beta1_t,
            self._adamw_beta2_t, self._adamw_eps_t, self._adamw_wd_t,
        )

        state['exp_avg'].copy_(exp_avg_g, non_blocking=True)
        state['exp_avg_sq'].copy_(exp_avg_sq_g, non_blocking=True)
        # Drop the per-param GPU transient so the allocator can return
        # its block to the driver in the empty_cache() at function exit.
        del exp_avg_g, exp_avg_sq_g

    # empty_cache at end: same rationale as patched_step_muon — release
    # the AdamW transient blocks back to the driver so the next 256
    # fwd+bwd cycles get a clean allocator.
    torch.cuda.empty_cache()


def patched_step_muon(self, group):
    """CPU-offloaded version of MuonAdamW._step_muon (single-GPU path)."""
    from nanochat.optim import muon_step_fused

    params = group['params']
    if not params:
        return

    p = params[0]
    state = self.state[p]
    num_params = len(params)
    shape, device, dtype = p.shape, p.device, p.dtype

    if "momentum_buffer" not in state:
        state["momentum_buffer"] = torch.zeros(
            num_params, *shape, dtype=dtype, device="cpu", pin_memory=True,
        )
    if "second_momentum_buffer" not in state:
        state_shape = (
            (num_params, shape[-2], 1)
            if shape[-2] >= shape[-1]
            else (num_params, 1, shape[-1])
        )
        state["second_momentum_buffer"] = torch.zeros(
            state_shape, dtype=dtype, device="cpu", pin_memory=True,
        )
    red_dim = -1 if shape[-2] >= shape[-1] else -2

    # Single-GPU's muon H2D copies the FULL state (no ZeRO sharding to
    # only-copy-owned-slice), so we briefly need ~5 GiB of contiguous GPU
    # space. The allocator cache from the preceding 256 fwd+bwd cycles
    # holds lots of small/medium blocks that cudaMalloc can't merge into
    # a 5 GiB request → OOM with reserved-but-unallocated. Returning the
    # cache to the driver here costs ~tens of ms (cudaFree pool) but lets
    # the big H2D find contiguous memory. Cheap vs the ~131 s single-GPU
    # iter time.
    torch.cuda.empty_cache()

    # H2D copy full state to GPU buffers
    mom_g = state["momentum_buffer"].to(device, non_blocking=True)
    mom2_g = state["second_momentum_buffer"].to(device, non_blocking=True)

    stacked_grads = torch.stack([p.grad for p in params])
    stacked_params = torch.stack(params)

    self._muon_momentum_t.fill_(group["momentum"])
    self._muon_beta2_t.fill_(group["beta2"] if group["beta2"] is not None else 0.0)
    self._muon_lr_t.fill_(group["lr"] * max(1.0, shape[-2] / shape[-1]) ** 0.5)
    self._muon_wd_t.fill_(group["weight_decay"])

    muon_step_fused(
        stacked_grads, stacked_params,
        mom_g, mom2_g,
        self._muon_momentum_t, self._muon_lr_t, self._muon_wd_t, self._muon_beta2_t,
        group["ns_steps"], red_dim,
    )

    # D2H copy state back
    state["momentum_buffer"].copy_(mom_g, non_blocking=True)
    state["second_momentum_buffer"].copy_(mom2_g, non_blocking=True)

    # Copy updated params back to originals (same as original implementation)
    torch._foreach_copy_(params, list(stacked_params.unbind(0)))

    # Free the function's GPU transients (mom_g, mom2_g, stacked_grads,
    # stacked_params, ~10+ GiB combined) back to the driver. Without this,
    # the allocator caches them as huge contiguous blocks that can't be
    # split for the next iter's many small activation allocs → effective
    # GPU usage stays high across the optim step boundary. With this,
    # the next 256 fwd+bwd cycles get a clean allocator. Costs ~tens of
    # ms per optim step (once per training iter, so ~0.05% on a ~130 s
    # single-GPU iter).
    del mom_g, mom2_g, stacked_grads, stacked_params
    torch.cuda.empty_cache()
