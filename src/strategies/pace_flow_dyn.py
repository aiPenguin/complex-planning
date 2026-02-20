"""
PACE-Flow-Dyn unmasking strategy shared by LLaDA and Dream.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Tuple

import torch
import torch.nn.functional as F

from src.utils.dream_utils import sample_tokens


# Token-level state machine:
# M = masked (to be predicted), P = provisional (candidate), F = frozen (committed).
_STATE_MASKED = 0
_STATE_PROVISIONAL = 1
_STATE_FROZEN = 2


class _PaceMasks:
    """Mask helpers for eligible and generation spans."""

    @staticmethod
    def eligible_mask(
        *,
        batch_size: int,
        seq_len: int,
        prompt_len: int,
        block_end: int | None,
        block_length: int,
        device: torch.device,
    ) -> torch.Tensor:
        # Only eligible tokens can change state this step.
        eligible = torch.zeros((batch_size, seq_len), dtype=torch.bool, device=device)
        if block_end is not None:
            start = max(prompt_len, block_end - block_length)
            end = min(block_end, seq_len)
            if start < end:
                eligible[:, start:end] = True
        else:
            if prompt_len < seq_len:
                eligible[:, prompt_len:seq_len] = True
        return eligible

    @staticmethod
    def generation_mask(
        *,
        batch_size: int,
        seq_len: int,
        prompt_len: int,
        device: torch.device,
    ) -> torch.Tensor:
        # Generation span excludes prompt tokens.
        mask = torch.zeros((batch_size, seq_len), dtype=torch.bool, device=device)
        if prompt_len < seq_len:
            mask[:, prompt_len:seq_len] = True
        return mask


class _PaceSchedule:
    """Progress and scheduling utilities."""

    @staticmethod
    def progress(global_step: int, total_steps: int) -> float:
        # Normalized step in [0, 1].
        if total_steps <= 1:
            return 1.0
        return min(1.0, float(global_step) / float(total_steps - 1))

    @staticmethod
    def value(progress: float, kind: str) -> float:
        # Supported schedules: linear, cosine, and *_decay variants.
        if kind in ("none", "constant", None):
            return 1.0
        invert = False
        if kind.endswith("_decay"):
            invert = True
            kind = kind[: -len("_decay")]
        if kind == "linear":
            base = progress
        elif kind == "quadratic":
            base = progress * progress
        elif kind == "sqrt":
            base = math.sqrt(progress)
        elif kind == "smoothstep":
            base = progress * progress * (3.0 - 2.0 * progress)
        elif kind == "cosine":
            base = 0.5 - 0.5 * math.cos(math.pi * progress)
        elif kind == "sigmoid":
            # Symmetric S-curve on [0, 1].
            x = 12.0 * (progress - 0.5)
            base = 1.0 / (1.0 + math.exp(-x))
        else:
            raise ValueError(f"Unknown schedule: {kind}")
        return 1.0 - base if invert else base


class _PaceStats:
    """Voting stats and masked ranking helpers."""

    @staticmethod
    def vote_stats(
        tokens: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # Compute variation entropy and agreement statistics per position.
        batch, num_particles, seq_len = tokens.shape
        ve = torch.zeros((batch, seq_len), device=tokens.device, dtype=torch.float32)
        consensus = torch.zeros_like(ve)
        pairwise = torch.zeros_like(ve)
        mode_tokens = torch.zeros((batch, seq_len), device=tokens.device, dtype=tokens.dtype)
        for b in range(batch):
            for i in range(seq_len):
                vals = tokens[b, :, i]
                unique, counts = vals.unique(return_counts=True)
                probs = counts.float() / num_particles
                ve[b, i] = -(probs * torch.log(probs)).sum()
                max_idx = torch.argmax(probs)
                consensus[b, i] = probs[max_idx]
                if num_particles > 1:
                    pairwise[b, i] = (counts.float() * (counts.float() - 1)).sum() / (
                        num_particles * (num_particles - 1)
                    )
                mode_tokens[b, i] = unique[max_idx]
        return ve, consensus, mode_tokens, pairwise

    @staticmethod
    def ranknorm(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # Rank-normalize within eligible positions.
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

    @staticmethod
    def masked_rank(
        values: torch.Tensor, mask: torch.Tensor, *, descending: bool
    ) -> torch.Tensor:
        # 1-based ranks over masked positions; used for RRF fusion.
        out = torch.zeros_like(values, dtype=torch.float32)
        for b in range(values.shape[0]):
            idx = mask[b].nonzero(as_tuple=False).squeeze(-1)
            if idx.numel() == 0:
                continue
            vals = values[b, idx]
            order = torch.argsort(vals, descending=descending)
            ranks = torch.empty_like(order, dtype=torch.float32)
            ranks[order] = torch.arange(
                1, order.numel() + 1, device=values.device, dtype=torch.float32
            )
            out[b, idx] = ranks
        return out

    @staticmethod
    def masked_quantile(
        values: torch.Tensor, mask: torch.Tensor, q: float
    ) -> torch.Tensor:
        # torch.quantile requires floating point inputs.
        out = torch.zeros(values.shape[0], device=values.device, dtype=torch.float32)
        for b in range(values.shape[0]):
            idx = mask[b].nonzero(as_tuple=False).squeeze(-1)
            if idx.numel() == 0:
                continue
            out[b] = torch.quantile(values[b, idx].float(), q)
        return out


class _PaceLogitControls:
    """Logit post-processing for sampling."""

    @staticmethod
    def apply(strategy: "PACEStrategy", logits: torch.Tensor) -> torch.Tensor:
        # Optionally suppress EOS/EOT logits.
        if strategy.logits_eos_inf or strategy.confidence_eos_eot_inf:
            logits = logits.clone()
            logits[:, :, :, strategy.eos_token_id] = -torch.inf
            if strategy.confidence_eos_eot_inf:
                logits[:, :, :, strategy.eot_token_id] = -torch.inf
        return logits

    @staticmethod
    def gumbelize(logits: torch.Tensor, temperature: float) -> torch.Tensor:
        if temperature == 0:
            return logits
        logits = logits.to(torch.float64)
        noise = torch.rand_like(logits, dtype=torch.float64)
        gumbel_noise = (-torch.log(noise)) ** temperature
        return logits.exp() / gumbel_noise


class _PaceTransitions:
    """State transition rules for M/P/F tokens."""

    @staticmethod
    def apply(
        strategy: "PACEStrategy",
        *,
        s_score: torch.Tensor,
        ve: torch.Tensor,
        agreement: torch.Tensor,
        g: torch.Tensor,
        eligible: torch.Tensor,
        is_last: bool,
        progress: float,
    ) -> None:
        # Operates in-place on strategy._token_state and lock trackers.
        if (
            strategy._token_state is None
            or strategy._last_freeze is None
            or strategy._last_provisional is None
        ):
            return

        token_state = strategy._token_state
        last_freeze = strategy._last_freeze
        last_provisional = strategy._last_provisional

        eligible_count_i = eligible.sum(dim=1).clamp_min(1)
        eligible_count = eligible_count_i.float()
        frozen_count = ((token_state == _STATE_FROZEN) & eligible).sum(dim=1).float()
        f = frozen_count / eligible_count

        masked_mask = (token_state == _STATE_MASKED) & eligible
        masked_count = masked_mask.sum(dim=1)
        target_mask_frac = 1.0 - _PaceSchedule.value(progress, strategy.mask_schedule)
        target_mask = torch.round(eligible_count * target_mask_frac).long()
        target_mask = torch.minimum(target_mask, eligible_count_i)
        target_mask = torch.clamp_min(target_mask, 0)

        # M -> P (rate-controlled to follow mask schedule)
        promote_count = (masked_count - target_mask).clamp_min(0)
        for b in range(token_state.shape[0]):
            k = int(promote_count[b].item())
            if k <= 0:
                continue
            scores = -g[b].clone()
            scores[~masked_mask[b]] = -torch.inf
            _, top_idx = torch.topk(scores, k=k)
            token_state[b, top_idx] = _STATE_PROVISIONAL
            last_provisional[b, top_idx] = strategy._global_step
        tau_base = _PaceSchedule.value(progress, strategy.tau_c_schedule)
        tau_step = strategy.tau_c_min + (strategy.tau_c_max - strategy.tau_c_min) * tau_base
        tau_step = torch.full_like(f, tau_step)
        if strategy.tau_c_f_weight > 0:
            tau_f = strategy.tau_c_min + (strategy.tau_c_max - strategy.tau_c_min) * f
            tau_c = (1 - strategy.tau_c_f_weight) * tau_step + strategy.tau_c_f_weight * tau_f
        else:
            tau_c = tau_step
        q_base = 1.0 - _PaceSchedule.value(progress, strategy.q_schedule)
        q_t = strategy.q0 * q_base * (1 - f)
        r_t = (strategy.r0 * (1 - f) * eligible_count).long()

        s_thresh = _PaceStats.masked_quantile(
            s_score, eligible, strategy.freeze_s_quantile
        )

        # P -> F (freeze when agreement/confidence are high enough)
        for b in range(token_state.shape[0]):
            freeze_mask = (
                (token_state[b] == _STATE_PROVISIONAL)
                & eligible[b]
                & (agreement[b] >= tau_c[b])
                & (s_score[b] >= s_thresh[b])
            )
            if strategy.provisional_lock > 0:
                freeze_mask = freeze_mask & (
                    (strategy._global_step - last_provisional[b]) >= strategy.provisional_lock
                )
            if freeze_mask.any().item():
                token_state[b, freeze_mask] = _STATE_FROZEN
                last_freeze[b, freeze_mask] = strategy._global_step

        # P -> M (remask a fraction of provisional tokens)
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

        # F -> M (rare unfreeze: send low-confidence frozen tokens back to masked)
        if strategy.r0 > 0 and not is_last:
            ve_hi = _PaceStats.masked_quantile(ve, eligible, strategy.ve_quantile_hi)
            s_lo = _PaceStats.masked_quantile(s_score, eligible, strategy.s_quantile_lo)
            for b in range(token_state.shape[0]):
                if r_t[b] <= 0:
                    continue
                cooldown = (strategy._global_step - last_freeze[b]) >= strategy.lock
                cand = (
                    (token_state[b] == _STATE_FROZEN)
                    & eligible[b]
                    & cooldown
                    & (ve[b] >= ve_hi[b])
                    & (s_score[b] <= s_lo[b])
                )
                if not cand.any().item():
                    continue
                scores = ve[b].clone()
                scores[~cand] = -torch.inf
                _, top_idx = torch.topk(
                    scores, k=min(r_t[b].item(), cand.sum().item())
                )
                token_state[b, top_idx] = _STATE_MASKED

        if is_last:
            final_mask = (token_state == _STATE_PROVISIONAL) & eligible
            if final_mask.any().item():
                token_state[final_mask] = _STATE_FROZEN
                last_freeze[final_mask] = strategy._global_step


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
    q_schedule: str = "linear"
    tau_c_schedule: str = "linear"
    mask_schedule: str = "linear"
    tau_c_f_weight: float = 0.0
    freeze_s_quantile: float = 0.5
    provisional_lock: int = 1
    lock: int = 2
    r0: float = 0.01
    early_stop_ve_eps: float | None = None
    ve_quantile_hi: float = 0.9
    s_quantile_lo: float = 0.1
    agreement_metric: str = "consensus"
    use_margin_confidence: bool = False
    margin_weight: float = 1.0
    rrf_k: float = 60.0
    use_noise: bool = True
    sampling_method: str = "sample"

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
    log_candidates_filename: str = "pace_candidates.pt"

    # Internal state (initialized via reset_state)
    _token_state: torch.Tensor | None = field(default=None, init=False, repr=False)
    _last_freeze: torch.Tensor | None = field(default=None, init=False, repr=False)
    _last_provisional: torch.Tensor | None = field(default=None, init=False, repr=False)
    _batch_size: int | None = field(default=None, init=False, repr=False)
    _seq_len: int | None = field(default=None, init=False, repr=False)
    _prompt_len: int | None = field(default=None, init=False, repr=False)
    _global_step: int = field(default=0, init=False, repr=False)
    _total_steps: int = field(default=1, init=False, repr=False)
    _last_mode_tokens: torch.Tensor | None = field(default=None, init=False, repr=False)
    _last_mode_score: torch.Tensor | None = field(default=None, init=False, repr=False)
    _last_particle_scores: torch.Tensor | None = field(default=None, init=False, repr=False)
    _last_mean_ve: torch.Tensor | None = field(default=None, init=False, repr=False)
    _last_primary_tokens: torch.Tensor | None = field(default=None, init=False, repr=False)
    _last_secondary_tokens: torch.Tensor | None = field(default=None, init=False, repr=False)
    _last_primary_source: list[str] | None = field(default=None, init=False, repr=False)
    _last_secondary_source: list[str] | None = field(default=None, init=False, repr=False)
    _last_particle_tokens: torch.Tensor | None = field(default=None, init=False, repr=False)

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
        self._global_step = 0
        self._total_steps = max(int(total_steps), 1)

        # Start with all tokens masked; prompt tokens are frozen immediately.
        token_state = torch.full(
            (batch_size, seq_len),
            _STATE_MASKED,
            dtype=torch.int64,
            device=device,
        )
        if prompt_len > 0:
            token_state[:, :prompt_len] = _STATE_FROZEN

        self._token_state = token_state
        # Track when positions last changed state to enforce lock periods.
        self._last_freeze = torch.full_like(token_state, -10**9)
        self._last_provisional = torch.full_like(token_state, -10**9)
        if prompt_len > 0:
            self._last_freeze[:, :prompt_len] = 0

        self._last_mode_tokens = None
        self._last_mode_score = None
        self._last_particle_scores = None
        self._last_mean_ve = None
        self._last_primary_tokens = None
        self._last_secondary_tokens = None
        self._last_primary_source = None
        self._last_secondary_source = None
        self._last_particle_tokens = None
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
        """Update masked positions for one diffusion step."""
        if (
            self._token_state is None
            or self._batch_size is None
            or self._seq_len is None
            or self._prompt_len is None
        ):
            raise RuntimeError("PACEStrategy.reset_state() must be called before unmask().")

        mask_id = mask_token_id if mask_token_id is not None else self.mask_id
        if mask_id is None:
            raise ValueError("mask_id/mask_token_id must be provided.")

        batch = self._batch_size
        seq_len = self._seq_len
        num_particles = self.num_particles
        if x.shape[0] != batch * num_particles:
            raise ValueError(
                f"Expected batch {batch * num_particles}, got {x.shape[0]}."
            )

        # Reshape to (B, P, L, ...).
        x_tokens = x.view(batch, num_particles, seq_len)
        logits = logits.view(batch, num_particles, seq_len, -1)

        eligible = _PaceMasks.eligible_mask(
            batch_size=batch,
            seq_len=seq_len,
            prompt_len=self._prompt_len,
            block_end=block_end,
            block_length=self.block_length,
            device=x.device,
        )
        active = (
            (self._token_state == _STATE_MASKED)
            | (self._token_state == _STATE_PROVISIONAL)
        ) & eligible

        active_particles = active.unsqueeze(1).expand(batch, num_particles, seq_len)
        y_tokens = x_tokens.clone()

        temperature = self.temperature if self.use_noise else 0.0
        if active_particles.any().item():
            logits_to_sample = _PaceLogitControls.apply(self, logits)
            if self.sampling_method == "argmax":
                logits_to_sample = _PaceLogitControls.gumbelize(
                    logits_to_sample, temperature
                )
                x0_all = torch.argmax(logits_to_sample, dim=-1)
                y_tokens[active_particles] = x0_all[active_particles]
            elif self.sampling_method == "sample":
                flat_logits = logits_to_sample[active_particles]
                if flat_logits.numel() > 0:
                    _, sampled = sample_tokens(
                        flat_logits,
                        temperature=temperature,
                        top_p=self.top_p,
                        top_k=self.top_k,
                    )
                    y_tokens[active_particles] = sampled
            else:
                raise ValueError(f"Unknown sampling_method: {self.sampling_method}")

        # Per-particle log-probabilities for confidence and voting stats.
        log_probs = F.log_softmax(logits, dim=-1)
        particle_logprob = torch.gather(
            log_probs, dim=-1, index=y_tokens.unsqueeze(-1)
        ).squeeze(-1)
        s_mean = particle_logprob.mean(dim=1)

        if num_particles == 1:
            ve = torch.zeros_like(s_mean)
            agreement = torch.ones_like(s_mean)
            mode_tokens = y_tokens[:, 0]
        else:
            ve, consensus, mode_tokens, pairwise = _PaceStats.vote_stats(y_tokens)
            if self.agreement_metric == "pairwise":
                agreement = pairwise
            else:
                agreement = consensus

        # Confidence score used to promote/freeze/remask tokens.
        s_score = s_mean
        if self.use_margin_confidence:
            probs = log_probs.exp()
            top2 = probs.topk(2, dim=-1).values
            margin = top2[..., 0] - top2[..., 1]
            margin_mean = margin.mean(dim=1)
            s_score = s_score + self.margin_weight * margin_mean

        self._last_mean_ve = (
            (ve * eligible).sum() / (eligible.sum().clamp_min(1))
        )

        # Joint ranking score: higher g => lower confidence / higher uncertainty.
        g = _PaceStats.ranknorm(-s_score, eligible)
        if num_particles > 1:
            r1 = _PaceStats.masked_rank(-s_score, eligible, descending=True)
            r2 = _PaceStats.masked_rank(ve, eligible, descending=True)
            g = 1.0 / (self.rrf_k + r1) + 1.0 / (self.rrf_k + r2)

        is_last = (steps is not None and step is not None and step >= steps - 1)
        progress = _PaceSchedule.progress(self._global_step, self._total_steps)
        _PaceTransitions.apply(
            self,
            s_score=s_score,
            ve=ve,
            agreement=agreement,
            g=g,
            eligible=eligible,
            is_last=is_last,
            progress=progress,
        )

        # Remask tokens still in M state for the next diffusion step.
        next_tokens = y_tokens.clone()
        remask = self._token_state == _STATE_MASKED
        if remask.any().item():
            remask_particles = remask.unsqueeze(1).expand(
                batch, num_particles, seq_len
            )
            next_tokens[remask_particles] = mask_id
        next_tokens = next_tokens.view(batch * num_particles, seq_len)

        # Cache scores for finalize() selection across particles.
        self._update_finalize_scores(
            log_probs=log_probs,
            y_tokens=y_tokens,
            mode_tokens=mode_tokens,
            gen_mask=_PaceMasks.generation_mask(
                batch_size=batch,
                seq_len=seq_len,
                prompt_len=self._prompt_len,
                device=x.device,
            ),
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

    def consume_particle_log(self) -> torch.Tensor | None:
        """Return the current batch particle log and clear it."""
        if self.batch_particle_log is None:
            return None
        log = self.batch_particle_log
        self.batch_particle_log = None
        return log

    def consume_generation_candidates(self) -> dict | None:
        """Return last primary/secondary/mode/best selections and clear them."""
        if self._last_primary_tokens is None:
            return None
        payload = {
            "primary": self._last_primary_tokens,
            "secondary": self._last_secondary_tokens,
            "primary_source": self._last_primary_source,
            "secondary_source": self._last_secondary_source,
            "particles": self._last_particle_tokens,
            "prompt_len": self._prompt_len,
            "seq_len": self._seq_len,
        }
        self._last_primary_tokens = None
        self._last_secondary_tokens = None
        self._last_primary_source = None
        self._last_secondary_source = None
        self._last_particle_tokens = None
        return payload

    def finalize(self, x: torch.Tensor) -> torch.Tensor:
        """Collapse particles into a single sequence per batch."""
        if self.num_particles == 1:
            tokens = x.view(self._batch_size, 1, self._seq_len)
            choose_mode = torch.ones((self._batch_size,), device=x.device, dtype=torch.bool)
            primary = tokens[:, 0]
            secondary = tokens[:, 0]
            self._last_primary_tokens = primary.detach()
            self._last_secondary_tokens = secondary.detach()
            self._last_primary_source = ["mode"] * self._batch_size
            self._last_secondary_source = ["best"] * self._batch_size
            self._last_particle_tokens = tokens.detach()
            return primary

        tokens = x.view(self._batch_size, self.num_particles, self._seq_len)

        if (
            self._last_particle_scores is None
            or self._last_mode_score is None
            or self._last_mode_tokens is None
        ):
            primary = tokens[:, 0]
            self._last_primary_tokens = primary.detach()
            self._last_secondary_tokens = None
            self._last_primary_source = ["best"] * self._batch_size
            self._last_secondary_source = [None] * self._batch_size
            self._last_particle_tokens = tokens.detach()
            return primary

        best_idx = torch.argmax(self._last_particle_scores, dim=-1)
        best_tokens = tokens[torch.arange(self._batch_size, device=x.device), best_idx]

        choose_mode = self._last_mode_score >= self._last_particle_scores[
            torch.arange(self._batch_size, device=x.device), best_idx
        ]
        primary = best_tokens.clone()
        primary[choose_mode] = self._last_mode_tokens[choose_mode]
        secondary = self._last_mode_tokens.clone()
        secondary[choose_mode] = best_tokens[choose_mode]

        self._last_primary_tokens = primary.detach()
        self._last_secondary_tokens = secondary.detach()
        self._last_primary_source = ["mode" if flag else "best" for flag in choose_mode.tolist()]
        self._last_secondary_source = ["best" if flag else "mode" for flag in choose_mode.tolist()]
        self._last_particle_tokens = tokens.detach()

        return primary

    def _update_finalize_scores(
        self,
        *,
        log_probs: torch.Tensor,
        y_tokens: torch.Tensor,
        mode_tokens: torch.Tensor,
        gen_mask: torch.Tensor,
    ) -> None:
        # Compute per-particle and mode log-probability totals on generated tokens.
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
