#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import torch


# Token state encodings logged by PACE-Flow-Dyn.
STATE_MASKED = 0
STATE_PROVISIONAL = 1
STATE_FROZEN = 2


def _safe_torch_load(path: Path) -> torch.Tensor:
    """Best-effort torch.load compatible with older/newer PyTorch."""
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")
    except Exception:
        return torch.load(path, map_location="cpu", weights_only=False)


def _validate_log(log: torch.Tensor) -> Tuple[int, int, int, int]:
    """Validate expected particle log shape and return dimensions."""
    if not isinstance(log, torch.Tensor):
        raise TypeError(f"Expected torch.Tensor, got {type(log)}")
    if log.ndim != 5 or log.shape[-1] != 2:
        raise ValueError(
            "Expected log shape [batch, num_particles, gen_len, steps, 2], "
            f"got {tuple(log.shape)}"
        )
    batch, num_particles, gen_len, steps, _ = log.shape
    if num_particles < 1 or gen_len < 1 or steps < 1:
        raise ValueError(f"Invalid log dimensions: {log.shape}")
    return batch, num_particles, gen_len, steps


def _state_fractions(states: np.ndarray) -> Dict[str, Dict[str, list]]:
    """Compute mean/variance of per-batch state fractions across positions."""
    steps = states.shape[-1]
    stats: Dict[str, Dict[str, list]] = {
        "masked": {"mean": [], "var": []},
        "provisional": {"mean": [], "var": []},
        "frozen": {"mean": [], "var": []},
        "valid": {"mean": [], "var": []},
    }
    for step in range(steps):
        st = states[:, :, step]
        valid = st != -1
        denom = valid.sum(axis=1)  # per-batch
        if denom.size == 0:
            for key in stats:
                stats[key]["mean"].append(float("nan"))
                stats[key]["var"].append(float("nan"))
            continue
        valid_frac = np.divide(
            denom.astype(np.float64),
            st.shape[1],
            out=np.full_like(denom, np.nan, dtype=np.float64),
            where=denom > 0,
        )
        for label, value in (
            ("masked", STATE_MASKED),
            ("provisional", STATE_PROVISIONAL),
            ("frozen", STATE_FROZEN),
        ):
            num = (st == value).sum(axis=1)
            frac = np.divide(
                num.astype(np.float64),
                denom,
                out=np.full_like(num, np.nan, dtype=np.float64),
                where=denom > 0,
            )
            stats[label]["mean"].append(float(np.nanmean(frac)))
            stats[label]["var"].append(float(np.nanvar(frac)))
        stats["valid"]["mean"].append(float(np.nanmean(valid_frac)))
        stats["valid"]["var"].append(float(np.nanvar(valid_frac)))
    return stats


def _first_step(states: np.ndarray, predicate: int | None) -> np.ndarray:
    """First step index where predicate matches (or first unmask if None)."""
    batch, gen_len, steps = states.shape
    out = np.full((batch, gen_len), -1, dtype=np.int64)
    for b in range(batch):
        for i in range(gen_len):
            st = states[b, i]
            if predicate is None:
                idx = np.where(st != STATE_MASKED)[0]
            else:
                idx = np.where(st == predicate)[0]
            if idx.size > 0:
                out[b, i] = int(idx[0])
    return out


def _histogram(first_steps: np.ndarray, steps: int) -> Dict[str, Any]:
    """Histogram counts for first-step indices; unreached counts in `unreached`."""
    valid = first_steps[first_steps >= 0]
    hist = np.bincount(valid, minlength=steps) if valid.size else np.zeros(steps, dtype=int)
    return {
        "counts": hist.astype(int).tolist(),
        "unreached": int((first_steps < 0).sum()),
    }


def _consensus_stats(tokens: np.ndarray) -> Tuple[list, list, list]:
    """Consensus stats (mean/p10/p90) over batch+positions per step."""
    batch, num_particles, gen_len, steps = tokens.shape
    mean_cons = []
    p10_cons = []
    p90_cons = []
    for step in range(steps):
        vals = []
        for b in range(batch):
            for i in range(gen_len):
                tok = tokens[b, :, i, step]
                valid = tok != -1
                if not valid.any():
                    continue
                unique, counts = np.unique(tok[valid], return_counts=True)
                vals.append(float(counts.max() / valid.sum()))
        if not vals:
            mean_cons.append(float("nan"))
            p10_cons.append(float("nan"))
            p90_cons.append(float("nan"))
        else:
            mean_cons.append(float(np.mean(vals)))
            p10_cons.append(float(np.percentile(vals, 10)))
            p90_cons.append(float(np.percentile(vals, 90)))
    return mean_cons, p10_cons, p90_cons


def _token_change_rate(tokens: np.ndarray) -> list:
    """Fraction of tokens that changed vs previous step (batch mean)."""
    batch, num_particles, gen_len, steps = tokens.shape
    rates = [0.0]
    for step in range(1, steps):
        cur = tokens[:, :, :, step]
        prev = tokens[:, :, :, step - 1]
        valid = (cur != -1) & (prev != -1)
        total = int(valid.sum())
        if total == 0:
            rates.append(float("nan"))
            continue
        changed = (cur != prev) & valid
        rates.append(float(changed.sum() / total))
    if len(rates) < steps:
        rates.extend([float("nan")] * (steps - len(rates)))
    return rates


def _consensus_heatmap(tokens: np.ndarray) -> np.ndarray:
    """Per-position consensus heatmap averaged over batch."""
    batch, num_particles, gen_len, steps = tokens.shape
    per_batch = np.full((batch, gen_len, steps), np.nan, dtype=np.float32)
    for b in range(batch):
        for step in range(steps):
            for i in range(gen_len):
                tok = tokens[b, :, i, step]
                valid = tok != -1
                if not valid.any():
                    continue
                _, counts = np.unique(tok[valid], return_counts=True)
                per_batch[b, i, step] = float(counts.max() / valid.sum())
    return np.nanmean(per_batch, axis=0)


def _state_heatmap(states: np.ndarray) -> np.ndarray:
    """Per-position state heatmap averaged over batch (0..2 expected values)."""
    out = states.copy().astype(np.float32)
    out[out < 0] = np.nan
    return np.nanmean(out, axis=0)


def _state_prob_heatmaps(states: np.ndarray) -> Dict[str, np.ndarray]:
    """Per-position probability heatmaps for each state (batch mean)."""
    out: Dict[str, np.ndarray] = {}
    for label, value in (
        ("masked", STATE_MASKED),
        ("provisional", STATE_PROVISIONAL),
        ("frozen", STATE_FROZEN),
    ):
        mask = states == value
        valid = states != -1
        denom = valid.sum(axis=0).astype(np.float32)
        num = mask.sum(axis=0).astype(np.float32)
        frac = np.divide(num, denom, out=np.full_like(num, np.nan), where=denom > 0)
        out[label] = frac
    return out


def _ensure_out_dir(path: Path) -> Path:
    """Create output directory if needed."""
    path.mkdir(parents=True, exist_ok=True)
    return path


def _maybe_import_matplotlib():
    """Try importing matplotlib for headless plotting; return success flag."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt  # noqa: F401
        return True
    except Exception:
        return False


def _write_plots(
    out_dir: Path,
    steps: int,
    dims: Tuple[int, int, int],
    state_fracs: Dict[str, Dict[str, list]],
    consensus_mean: list,
    consensus_p10: list,
    consensus_p90: list,
    change_rate: list,
    freeze_hist: Dict[str, Any],
    unmask_hist: Dict[str, Any],
    consensus_heat: np.ndarray | None,
    state_heat: np.ndarray | None,
    state_probs: Dict[str, np.ndarray] | None,
) -> None:
    """Write all plots to disk using matplotlib."""
    import matplotlib.pyplot as plt

    x = np.arange(steps)
    batch, num_particles, gen_len = dims

    # State fractions
    plt.figure(figsize=(8, 4))
    for label in ("masked", "provisional", "frozen"):
        mean = np.array(state_fracs[label]["mean"], dtype=np.float64)
        var = np.array(state_fracs[label]["var"], dtype=np.float64)
        std = np.sqrt(np.maximum(var, 0.0))
        plt.plot(x, mean, label=f"{label} mean")
        plt.fill_between(x, mean - std, mean + std, alpha=0.2, label=f"{label} ±1 std")
    plt.ylim(0.0, 1.0)
    plt.xlabel("step")
    plt.ylabel("fraction")
    plt.title(
        f"Token State Fractions (mean ± std)  B={batch} P={num_particles} G={gen_len}"
    )
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "state_fractions.png", dpi=160)
    plt.close()

    # Consensus + change rate
    plt.figure(figsize=(8, 4))
    plt.plot(x, consensus_mean, label="consensus_mean")
    plt.fill_between(x, consensus_p10, consensus_p90, alpha=0.2, label="consensus p10-p90")
    plt.plot(x, change_rate, label="token_change_rate")
    plt.ylim(0.0, 1.0)
    plt.xlabel("step")
    plt.ylabel("fraction")
    plt.title(
        f"Consensus and Token Change Rate (mean over batch)  B={batch} P={num_particles} G={gen_len}"
    )
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "consensus_change.png", dpi=160)
    plt.close()

    # Freeze/unmask histograms
    plt.figure(figsize=(8, 4))
    plt.bar(x, freeze_hist["counts"], alpha=0.6, label="first_freeze")
    plt.bar(x, unmask_hist["counts"], alpha=0.6, label="first_unmask")
    plt.xlabel("step")
    plt.ylabel("count")
    plt.title(
        f"First Freeze/Unmask Step Histogram  B={batch} P={num_particles} G={gen_len}"
    )
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "first_step_hist.png", dpi=160)
    plt.close()

    # Consensus heatmap
    if consensus_heat is not None:
        plt.figure(figsize=(9, 4))
        plt.imshow(
            consensus_heat,
            aspect="auto",
            origin="lower",
            vmin=0.0,
            vmax=1.0,
            cmap="viridis",
        )
        plt.colorbar(label="consensus")
        plt.xlabel("step")
        plt.ylabel("gen position")
        plt.title("Consensus Heatmap (mean over batch)")
        plt.tight_layout()
        plt.savefig(out_dir / "consensus_heatmap.png", dpi=160)
        plt.close()

    # State heatmap
    if state_heat is not None:
        plt.figure(figsize=(9, 4))
        cmap = plt.get_cmap("tab10", 3)
        plt.imshow(
            state_heat,
            aspect="auto",
            origin="lower",
            vmin=0.0,
            vmax=2.0,
            cmap=cmap,
        )
        cbar = plt.colorbar(ticks=[0, 1, 2])
        cbar.ax.set_yticklabels(["masked", "provisional", "frozen"])
        plt.xlabel("step")
        plt.ylabel("gen position")
        plt.title("Token State Heatmap (mean over batch)")
        plt.tight_layout()
        plt.savefig(out_dir / "state_heatmap.png", dpi=160)
        plt.close()

    if state_probs is not None:
        fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharey=True, constrained_layout=True)
        for ax, label in zip(axes, ("masked", "provisional", "frozen")):
            img = ax.imshow(
                state_probs[label],
                aspect="auto",
                origin="lower",
                vmin=0.0,
                vmax=1.0,
                cmap="viridis",
            )
            ax.set_title(f"{label} fraction")
            ax.set_xlabel("step")
        axes[0].set_ylabel("gen position")
        fig.colorbar(img, ax=axes, shrink=0.85, pad=0.02, label="fraction")
        fig.suptitle("Token State Fractions by Position (mean over batch)")
        fig.savefig(out_dir / "state_prob_heatmaps.png", dpi=160)
        plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Analyze and visualize PACE particle logs."
    )
    parser.add_argument("--log", required=True, type=Path, help="Path to pace_particles.pt")
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output directory (default: <log_dir>/particle_analysis)",
    )
    parser.add_argument(
        "--batch-index",
        type=int,
        default=None,
        help="Optional single batch index to analyze",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip plot generation (JSON summary only)",
    )
    args = parser.parse_args()

    log = _safe_torch_load(args.log)
    batch, num_particles, gen_len, steps = _validate_log(log)

    if args.batch_index is not None:
        if args.batch_index < 0 or args.batch_index >= batch:
            raise ValueError(f"batch-index {args.batch_index} out of range [0, {batch - 1}]")
        log = log[args.batch_index : args.batch_index + 1]
        batch = 1

    states = log[..., 0].cpu().numpy()  # [B, P, G, S]
    tokens = log[..., 1].cpu().numpy()  # [B, P, G, S]
    states0 = states[:, 0, :, :]  # token state identical across particles

    state_fracs = _state_fractions(states0)
    first_freeze = _first_step(states0, STATE_FROZEN)
    first_unmask = _first_step(states0, None)
    freeze_hist = _histogram(first_freeze, steps)
    unmask_hist = _histogram(first_unmask, steps)

    consensus_mean, consensus_p10, consensus_p90 = _consensus_stats(tokens)
    change_rate = _token_change_rate(tokens)

    summary: Dict[str, Any] = {
        "log_path": str(args.log),
        "dimensions": {
            "batch": batch,
            "num_particles": num_particles,
            "gen_len": gen_len,
            "steps": steps,
        },
        "averaging": "per-batch mean/variance across positions",
        "state_fractions": state_fracs,
        "consensus": {
            "mean": consensus_mean,
            "p10": consensus_p10,
            "p90": consensus_p90,
        },
        "token_change_rate": change_rate,
        "first_freeze_hist": freeze_hist,
        "first_unmask_hist": unmask_hist,
        "field_notes": {
            "dimensions": "Tensor dimensions after optional batch slicing.",
            "state_fractions": "Per-step mean/variance of per-batch state fractions.",
            "consensus": "Per-step consensus stats over batch+positions.",
            "token_change_rate": "Per-step fraction of tokens that changed vs previous step.",
            "first_freeze_hist": "Histogram of first step when a position becomes frozen.",
            "first_unmask_hist": "Histogram of first step when a position is no longer masked.",
        },
    }

    out_dir = args.out
    if out_dir is None:
        out_dir = args.log.parent / "particle_analysis"
    out_dir = _ensure_out_dir(out_dir)

    summary_path = out_dir / "particle_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))

    if args.no_plots:
        print(f"Wrote summary to {summary_path}")
        return 0

    if not _maybe_import_matplotlib():
        print("matplotlib not available; wrote JSON summary only.")
        print(f"Wrote summary to {summary_path}")
        return 0

    consensus_heat = _consensus_heatmap(tokens) if batch > 0 else None
    state_heat = _state_heatmap(states0) if batch > 0 else None
    state_probs = _state_prob_heatmaps(states0) if batch > 0 else None
    _write_plots(
        out_dir=out_dir,
        steps=steps,
        dims=(batch, num_particles, gen_len),
        state_fracs=state_fracs,
        consensus_mean=consensus_mean,
        consensus_p10=consensus_p10,
        consensus_p90=consensus_p90,
        change_rate=change_rate,
        freeze_hist=freeze_hist,
        unmask_hist=unmask_hist,
        consensus_heat=consensus_heat,
        state_heat=state_heat,
        state_probs=state_probs,
    )

    print(f"Wrote summary and plots to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
