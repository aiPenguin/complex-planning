"""
PACE-Flow-Dyn unmasking strategy shared by LLaDA and Dream.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math
from pathlib import Path
from typing import Tuple

import torch
import torch.nn.functional as F

from src.utils.dream_utils import sample_tokens


_STATE_MASKED = 0
_STATE_PROVISIONAL = 1
_STATE_FROZEN = 2


@dataclass
class PACEStrategy:
    """Shared PACE-Flow-Dyn strategy for LLaDA and Dream diffusion loops."""

    # Common decoding knobs
    steps: int = 128
    max_new_tokens: int | None = 128
    gen_length: int | None = None
    block_length: int = 32
    temperature: float = 0.0
    cfg_scale: float = 0.0
    mask_id: int = 126336

    # Dream-related knobs (kept for compatibility)
    top_p: float | None = 0.95
    top_k: int | None = None
    alg: str = "pace_flow_dyn"
    alg_temp: float | None = 0.0
    eps: float = 1e-3

    # PACE-Flow-Dyn hyperparameters
    num_particles: int = 4
    q0: float = 0.15
    tau_c_min: float = 0.70
    tau_c_max: float = 0.90
    lock: int = 2
    r0: float = 0.01
    early_stop_ve_eps: float | None = None
    ve_quantile_hi: float = 0.9
    s_quantile_lo: float = 0.1

    # LLaDA-specific compatibility flags
    eos_token_id: int = 126081
    eot_token_id: int = 126348
    logits_eos_inf: bool = False
    confidence_eos_eot_inf: bool = False
    
    supports_pace_flow_dyn: bool = field(default=True, init=False)

    # Log
    batch_particle_log: torch.Tensor | None = field(default=None, init=False, repr=False)
    log_particles: bool = False
    log_particles_to_cpu: bool = True
    log_particles_filename: str = "pace_particles.pt"

    # Internal state (initialized via reset_state)
    _token_state: torch.Tensor | None = field(default=None, init=False, repr=False)
    _last_freeze: torch.Tensor | None = field(default=None, init=False, repr=False)
    _batch_size: int | None = field(default=None, init=False, repr=False)
    _seq_len: int | None = field(default=None, init=False, repr=False)
    _prompt_len: int | None = field(default=None, init=False, repr=False)
    _mask_id: int | None = field(default=None, init=False, repr=False)
    _global_step: int = field(default=0, init=False, repr=False)
    _last_mode_tokens: torch.Tensor | None = field(default=None, init=False, repr=False)
    _last_mode_score: torch.Tensor | None = field(default=None, init=False, repr=False)
    _last_particle_scores: torch.Tensor | None = field(default=None, init=False, repr=False)
    _last_mean_ve: torch.Tensor | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.max_new_tokens is None and self.gen_length is None:
            raise ValueError("PACEStrategy requires max_new_tokens or gen_length.")
        if self.max_new_tokens is None:
            self.max_new_tokens = self.gen_length
        if self.gen_length is None:
            self.gen_length = self.max_new_tokens
        if self.num_particles < 1:
            raise ValueError("num_particles must be >= 1")

    def reset_state(
        self,
        *,
        batch_size: int,
        seq_len: int,
        prompt_len: int,
        mask_id: int,
        device: torch.device,
        total_steps: int,
    ) -> None:
        """Initialize token state for a new generation run."""
        self._batch_size = batch_size
        self._seq_len = seq_len
        self._prompt_len = prompt_len
        self._mask_id = mask_id
        self._global_step = 0

        token_state = torch.full(
            (batch_size, seq_len),
            _STATE_MASKED,
            dtype=torch.int64,
            device=device,
        )
        if prompt_len > 0:
            token_state[:, :prompt_len] = _STATE_FROZEN

        self._token_state = token_state
        self._last_freeze = torch.full_like(token_state, -10**9)
        if prompt_len > 0:
            self._last_freeze[:, :prompt_len] = 0

        self._last_mode_tokens = None
        self._last_mode_score = None
        self._last_particle_scores = None
        self._last_mean_ve = None
        if self.log_particles:
            log_device = torch.device("cpu") if self.log_particles_to_cpu else device
            gen_length = max(seq_len - prompt_len, 0)
            self.batch_particle_log = torch.full(
                (batch_size, self.num_particles, gen_length, total_steps, 2),
                -1,
                dtype=torch.int64,
                device=log_device,
            )
        else:
            self.batch_particle_log = None

    def should_stop(self) -> bool:
        """Check if all generation tokens are frozen or VE is low."""
        if self._token_state is None or self._prompt_len is None:
            return False
        gen_state = self._token_state[:, self._prompt_len :]
        if gen_state.numel() == 0:
            return True
        if torch.all(gen_state == _STATE_FROZEN).item():
            return True
        if self.early_stop_ve_eps is not None and self._last_mean_ve is not None:
            return (self._last_mean_ve <= self.early_stop_ve_eps).item()
        return False

    def unmask(
        self,
        *,
        x: torch.Tensor,
        logits: torch.Tensor,
        mask_index: torch.Tensor,
        block_end: int | None = None,
        num_transfer_tokens: torch.Tensor | None = None,
        t: torch.Tensor | None = None,
        s: torch.Tensor | None = None,
        step: int | None = None,
        steps: int | None = None,
        mask_token_id: int | None = None,
    ) -> torch.Tensor:
        """Update masked positions for one diffusion step (LLaDA or Dream)."""
        if self._token_state is None or self._batch_size is None or self._seq_len is None:
            raise RuntimeError("PACEStrategy.reset_state() must be called before unmask().")

        mode = "llada" if block_end is not None else "dream"
        mask_id = self.mask_id if mode == "llada" else mask_token_id
        if mask_id is None:
            raise ValueError("mask_id/mask_token_id must be provided.")

        batch = self._batch_size
        seq_len = self._seq_len
        num_particles = self.num_particles
        if x.shape[0] != batch * num_particles:
            raise ValueError(
                f"Expected batch {batch * num_particles}, got {x.shape[0]}."
            )

        x_tokens = x.view(batch, num_particles, seq_len)
        logits = logits.view(batch, num_particles, seq_len, -1)

        eligible = self._eligible_mask(block_end, device=x.device)
        active = (self._token_state == _STATE_MASKED) & eligible

        active_particles = active.unsqueeze(1).expand(batch, num_particles, seq_len)
        y_tokens = x_tokens.clone()

        if mode == "llada" and active_particles.any().item():
            logits_to_sample = self._apply_llada_logit_controls(logits)
            x0_all = torch.argmax(logits_to_sample, dim=-1)
            y_tokens[active_particles] = x0_all[active_particles]
        elif mode == "dream" and active_particles.any().item():
            flat_logits = logits[active_particles]
            if flat_logits.numel() > 0:
                _, sampled = sample_tokens(
                    flat_logits,
                    temperature=self.temperature,
                    top_p=self.top_p,
                    top_k=self.top_k,
                )
                y_tokens[active_particles] = sampled

        log_probs = F.log_softmax(logits, dim=-1)
        particle_logprob = torch.gather(
            log_probs, dim=-1, index=y_tokens.unsqueeze(-1)
        ).squeeze(-1)
        s_mean = particle_logprob.mean(dim=1)

        if num_particles == 1:
            ve = torch.zeros_like(s_mean)
            consensus = torch.ones_like(s_mean)
            mode_tokens = y_tokens[:, 0]
        else:
            ve, consensus, mode_tokens = self._vote_stats(y_tokens)

        self._last_mean_ve = (
            (ve * eligible).sum() / (eligible.sum().clamp_min(1))
        )

        g = self._ranknorm(-s_mean, eligible)
        if num_particles > 1:
            g = g + self._ranknorm(ve, eligible)

        is_last = (steps is not None and step is not None and step >= steps - 1)
        self._apply_transitions(
            s_mean=s_mean,
            ve=ve,
            consensus=consensus,
            g=g,
            eligible=eligible,
            is_last=is_last,
        )

        next_tokens = y_tokens.clone()
        remask = self._token_state == _STATE_MASKED
        if remask.any().item():
            remask_particles = remask.unsqueeze(1).expand(
                batch, num_particles, seq_len
            )
            next_tokens[remask_particles] = mask_id
        next_tokens = next_tokens.view(batch * num_particles, seq_len)

        self._update_finalize_scores(
            log_probs=log_probs,
            y_tokens=y_tokens,
            mode_tokens=mode_tokens,
            gen_mask=self._generation_mask(device=x.device),
        )

        if self.log_particles:
            step_idx = self._global_step
            if step_idx < self.batch_particle_log.shape[3]:
                gen_start = self._prompt_len
                gen_end = self._seq_len
                if gen_start < gen_end:
                    token_state = self._token_state[:, gen_start:gen_end]
                    token_state = token_state.unsqueeze(1).expand(
                        batch, num_particles, gen_end - gen_start
                    )
                    y_tokens_log = y_tokens[:, :, gen_start:gen_end].detach()
                    if self.log_particles_to_cpu:
                        token_state = token_state.cpu()
                        y_tokens_log = y_tokens_log.cpu()
                    self.batch_particle_log[:, :, :, step_idx, 0] = token_state
                    self.batch_particle_log[:, :, :, step_idx, 1] = y_tokens_log

        self._global_step += 1
        return next_tokens

    def save_particle_log(self, output_dir: str) -> str | None:
        """Save logged particle tokens to a .pt file in output_dir."""
        if not self.log_particles or self.batch_particle_log is None:
            return None
        path = Path(output_dir) / self.log_particles_filename
        torch.save(self.batch_particle_log, path)
        return str(path)

    def consume_particle_log(self) -> torch.Tensor | None:
        """Return the current batch particle log and clear it."""
        if self.batch_particle_log is None:
            return None
        log = self.batch_particle_log
        self.batch_particle_log = None
        return log

    def finalize(self, x: torch.Tensor) -> torch.Tensor:
        """Collapse particles into a single sequence per batch."""
        if self._batch_size is None or self._seq_len is None:
            return x
        if self.num_particles == 1:
            return x

        batch = self._batch_size
        seq_len = self._seq_len
        tokens = x.view(batch, self.num_particles, seq_len)

        if self._last_particle_scores is None or self._last_mode_tokens is None:
            best_idx = torch.zeros(batch, dtype=torch.long, device=x.device)
            return tokens[torch.arange(batch, device=x.device), best_idx]

        best_idx = torch.argmax(self._last_particle_scores, dim=-1)
        best_tokens = tokens[torch.arange(batch, device=x.device), best_idx]

        if self._last_mode_score is None:
            return best_tokens

        choose_mode = self._last_mode_score >= self._last_particle_scores[
            torch.arange(batch, device=x.device), best_idx
        ]
        out = best_tokens.clone()
        out[choose_mode] = self._last_mode_tokens[choose_mode]
        return out

    def _eligible_mask(self, block_end: int | None, device: torch.device) -> torch.Tensor:
        if self._prompt_len is None or self._seq_len is None:
            raise RuntimeError("reset_state must set prompt_len and seq_len.")
        prompt_len = self._prompt_len
        seq_len = self._seq_len
        eligible = torch.zeros((self._batch_size, seq_len), dtype=torch.bool, device=device)
        if block_end is not None:
            start = max(prompt_len, block_end - self.block_length)
            end = min(block_end, seq_len)
            if start < end:
                eligible[:, start:end] = True
        else:
            if prompt_len < seq_len:
                eligible[:, prompt_len:seq_len] = True
        return eligible

    def _generation_mask(self, device: torch.device) -> torch.Tensor:
        if self._prompt_len is None or self._seq_len is None:
            raise RuntimeError("reset_state must set prompt_len and seq_len.")
        mask = torch.zeros((self._batch_size, self._seq_len), dtype=torch.bool, device=device)
        if self._prompt_len < self._seq_len:
            mask[:, self._prompt_len : self._seq_len] = True
        return mask

    def _apply_llada_logit_controls(self, logits: torch.Tensor) -> torch.Tensor:
        if self.logits_eos_inf or self.confidence_eos_eot_inf:
            logits = logits.clone()
            logits[:, :, :, self.eos_token_id] = -torch.inf
            if self.confidence_eos_eot_inf:
                logits[:, :, :, self.eot_token_id] = -torch.inf
        if self.temperature == 0:
            return logits
        logits = logits.to(torch.float64)
        noise = torch.rand_like(logits, dtype=torch.float64)
        gumbel_noise = (-torch.log(noise)) ** self.temperature
        return logits.exp() / gumbel_noise

    def _vote_stats(
        self, tokens: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch, num_particles, seq_len = tokens.shape
        ve = torch.zeros((batch, seq_len), device=tokens.device, dtype=torch.float32)
        consensus = torch.zeros_like(ve)
        mode_tokens = torch.zeros((batch, seq_len), device=tokens.device, dtype=tokens.dtype)
        for b in range(batch):
            for i in range(seq_len):
                vals = tokens[b, :, i]
                unique, counts = vals.unique(return_counts=True)
                probs = counts.float() / num_particles
                ve[b, i] = -(probs * torch.log(probs)).sum()
                max_idx = torch.argmax(probs)
                consensus[b, i] = probs[max_idx]
                mode_tokens[b, i] = unique[max_idx]
        return ve, consensus, mode_tokens

    def _ranknorm(self, values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        out = torch.zeros_like(values, dtype=torch.float32)
        for b in range(values.shape[0]):
            idx = mask[b].nonzero(as_tuple=False).squeeze(-1)
            if idx.numel() == 0:
                continue
            vals = values[b, idx]
            order = torch.argsort(vals)
            ranks = torch.empty_like(order, dtype=torch.float32)
            ranks[order] = torch.arange(
                1, order.numel() + 1, device=values.device, dtype=torch.float32
            )
            out[b, idx] = (ranks - 0.5) / order.numel()
        return out

    def _masked_median(self, values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        med = torch.zeros(values.shape[0], device=values.device, dtype=values.dtype)
        for b in range(values.shape[0]):
            idx = mask[b].nonzero(as_tuple=False).squeeze(-1)
            if idx.numel() == 0:
                continue
            med[b] = values[b, idx].median()
        return med

    def _masked_quantile(
        self, values: torch.Tensor, mask: torch.Tensor, q: float
    ) -> torch.Tensor:
        # torch.quantile requires floating point inputs.
        out = torch.zeros(values.shape[0], device=values.device, dtype=torch.float32)
        for b in range(values.shape[0]):
            idx = mask[b].nonzero(as_tuple=False).squeeze(-1)
            if idx.numel() == 0:
                continue
            out[b] = torch.quantile(values[b, idx].float(), q)
        return out

    def _apply_transitions(
        self,
        *,
        s_mean: torch.Tensor,
        ve: torch.Tensor,
        consensus: torch.Tensor,
        g: torch.Tensor,
        eligible: torch.Tensor,
        is_last: bool,
    ) -> None:
        if self._token_state is None or self._last_freeze is None:
            return

        token_state = self._token_state

        # M -> P
        token_state[(token_state == _STATE_MASKED) & eligible] = _STATE_PROVISIONAL

        eligible_count = eligible.sum(dim=1).clamp_min(1).float()
        frozen_count = ((token_state == _STATE_FROZEN) & eligible).sum(dim=1).float()
        f = frozen_count / eligible_count
        tau_c = self.tau_c_min + (self.tau_c_max - self.tau_c_min) * f
        q_t = self.q0 * (1 - f)
        r_t = (self.r0 * (1 - f) * eligible_count).long()

        median_s = self._masked_median(s_mean, eligible)

        # P -> F
        for b in range(token_state.shape[0]):
            freeze_mask = (
                (token_state[b] == _STATE_PROVISIONAL)
                & eligible[b]
                & (consensus[b] >= tau_c[b])
                & (s_mean[b] >= median_s[b])
            )
            if freeze_mask.any().item():
                token_state[b, freeze_mask] = _STATE_FROZEN
                self._last_freeze[b, freeze_mask] = self._global_step

        # P -> M (remask)
        if not is_last:
            for b in range(token_state.shape[0]):
                candidate_mask = (token_state[b] == _STATE_PROVISIONAL) & eligible[b]
                num_candidates = candidate_mask.sum().item()
                if num_candidates == 0:
                    continue
                k = int(math.ceil(float(q_t[b]) * num_candidates))
                if k <= 0:
                    continue
                scores = g[b].clone()
                scores[~candidate_mask] = -torch.inf
                _, top_idx = torch.topk(scores, k=k)
                token_state[b, top_idx] = _STATE_MASKED

        # F -> P (rare unfreeze)
        if self.r0 > 0:
            ve_hi = self._masked_quantile(ve, eligible, self.ve_quantile_hi)
            s_lo = self._masked_quantile(s_mean, eligible, self.s_quantile_lo)
            for b in range(token_state.shape[0]):
                if r_t[b] <= 0:
                    continue
                cooldown = (self._global_step - self._last_freeze[b]) >= self.lock
                cand = (
                    (token_state[b] == _STATE_FROZEN)
                    & eligible[b]
                    & cooldown
                    & (ve[b] >= ve_hi[b])
                    & (s_mean[b] <= s_lo[b])
                )
                if not cand.any().item():
                    continue
                scores = ve[b].clone()
                scores[~cand] = -torch.inf
                _, top_idx = torch.topk(scores, k=min(r_t[b].item(), cand.sum().item()))
                token_state[b, top_idx] = _STATE_PROVISIONAL

    def _update_finalize_scores(
        self,
        *,
        log_probs: torch.Tensor,
        y_tokens: torch.Tensor,
        mode_tokens: torch.Tensor,
        gen_mask: torch.Tensor,
    ) -> None:
        batch = y_tokens.shape[0]
        num_particles = y_tokens.shape[1]
        seq_len = y_tokens.shape[2]

        if gen_mask.numel() == 0:
            return

        mean_log_probs = log_probs.mean(dim=1)
        mode_scores = torch.gather(
            mean_log_probs, dim=-1, index=mode_tokens.unsqueeze(-1)
        ).squeeze(-1)
        mode_scores = (mode_scores * gen_mask).sum(dim=-1)

        particle_scores = torch.gather(
            log_probs, dim=-1, index=y_tokens.unsqueeze(-1)
        ).squeeze(-1)
        particle_scores = particle_scores * gen_mask.unsqueeze(1)
        particle_scores = particle_scores.sum(dim=-1)

        self._last_mode_tokens = mode_tokens.detach()
        self._last_mode_score = mode_scores.detach()
        self._last_particle_scores = particle_scores.detach()


__all__ = ["PACEStrategy"]
