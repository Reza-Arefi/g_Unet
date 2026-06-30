#!/usr/bin/env python3
"""Split a combined u/v export into separate single-channel prefixes (copy mask/day/hour)."""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def split_uv_prefix(
    source_prefix: str,
    *,
    u_prefix: str | None = None,
    v_prefix: str | None = None,
    overwrite: bool = False,
) -> None:
    """Write ``{u,v}_prefix_*`` from ``source_prefix_sims.npy`` with C=2 (u,v)."""
    src = Path(source_prefix)
    sims_path = src.parent / f"{src.name}_sims.npy"
    if not sims_path.exists():
        raise FileNotFoundError(sims_path)

    sims = np.load(sims_path, mmap_mode="r")
    if sims.ndim != 5 or sims.shape[2] < 2:
        raise ValueError(f"Expected sims (N,T,2+,H,W), got {sims.shape}")

    sidecars = ("mask", "day", "hour", "latlon", "window", "indices")
    pairs: list[tuple[str, int]] = []
    if u_prefix:
        pairs.append((u_prefix, 0))
    if v_prefix:
        pairs.append((v_prefix, 1))
    if not pairs:
        raise ValueError("Set at least one of --u-prefix / --v-prefix")

    for out_prefix, ch_idx in pairs:
        out = Path(out_prefix)
        out.parent.mkdir(parents=True, exist_ok=True)
        out_sims = out.parent / f"{out.name}_sims.npy"
        if out_sims.exists() and not overwrite:
            print(f"Skip {out_sims} (exists; use --overwrite)")
            continue

        ch = np.asarray(sims[:, :, ch_idx : ch_idx + 1, :, :])
        np.save(out_sims, ch)
        print(f"Wrote {out_sims}  shape={ch.shape}")

        for name in sidecars:
            src_file = src.parent / f"{src.name}_{name}.npy"
            if not src_file.exists():
                continue
            dst_file = out.parent / f"{out.name}_{name}.npy"
            shutil.copy2(src_file, dst_file)
            print(f"  copied {dst_file.name}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Split u/v channels from a combined export prefix.")
    parser.add_argument(
        "source_prefix",
        help="e.g. data/sample_16_19_smallfixwin_uv_dt96",
    )
    parser.add_argument("--u-prefix", type=str, help="Output prefix for u-only sims")
    parser.add_argument("--v-prefix", type=str, help="Output prefix for v-only sims")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    split_uv_prefix(
        args.source_prefix,
        u_prefix=args.u_prefix,
        v_prefix=args.v_prefix,
        overwrite=args.overwrite,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
