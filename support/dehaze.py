#!/usr/bin/env python3
"""Recover scene radiance using refined transmission and DCP atmospheric light.

    I = J * t + A * (1 - t)
    J = (I - A) / max(t, t_min) + A

DCP atmospheric-light estimation follows He et al.:
https://people.csail.mit.edu/kaiming/publications/pami10dehaze.pdf
Only the atmospheric-light estimate uses DCP; transmission is loaded from disk.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np


IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}


def load_rgb(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Could not read image: {path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0


def load_transmission(path: Path, shape: tuple[int, int]) -> np.ndarray:
    transmission = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if transmission is None:
        raise ValueError(f"Could not read transmission: {path}")
    if transmission.ndim != 2 or transmission.dtype not in (np.uint8, np.uint16):
        raise ValueError(f"Expected an 8-bit or 16-bit grayscale transmission: {path}")
    transmission = transmission.astype(np.float32) / np.iinfo(transmission.dtype).max
    if transmission.shape != shape:
        transmission = cv2.resize(
            transmission, (shape[1], shape[0]), interpolation=cv2.INTER_LINEAR
        )
    return np.clip(transmission, 0.0, 1.0)


def estimate_atmospheric_light(
    hazy: np.ndarray, patch_size: int = 15, top_fraction: float = 0.001
) -> np.ndarray:
    """Choose one RGB pixel by summed intensity among bright dark-channel candidates."""
    if patch_size < 1 or patch_size % 2 != 1:
        raise ValueError("Dark-channel patch size must be a positive odd integer.")
    if not 0 < top_fraction <= 1:
        raise ValueError("Top fraction must be in (0, 1].")
    dark = cv2.erode(
        hazy.min(axis=2),
        np.ones((patch_size, patch_size), dtype=np.uint8),
        borderType=cv2.BORDER_REPLICATE,
    )
    count = max(1, int(dark.size * top_fraction))
    indices = np.argpartition(dark.ravel(), -count)[-count:]
    candidates = hazy.reshape(-1, 3)[indices]
    return candidates[np.argmax(candidates.sum(axis=1))].copy()


def recover_radiance(
    hazy: np.ndarray, transmission: np.ndarray, atmosphere: np.ndarray, t_min: float = 0.1
) -> np.ndarray:
    if not 0 < t_min <= 1:
        raise ValueError("Minimum transmission must be in (0, 1].")
    if hazy.shape != (*transmission.shape, 3) or atmosphere.shape != (3,):
        raise ValueError("Expected aligned RGB image, 2D transmission, and RGB atmosphere.")
    recovered = (hazy - atmosphere) / np.maximum(transmission[..., None], t_min) + atmosphere
    return np.clip(recovered, 0.0, 1.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hazy-dir", type=Path, default=Path("testdata/hazy"))
    parser.add_argument(
        "--trans-dir", type=Path, default=Path("testdata/tabpfn_reconstructed")
    )
    parser.add_argument("--output-dir", type=Path, default=Path("testdata/dehazed"))
    parser.add_argument(
        "--trans-suffix", default="_tabpfn.png",
        help="Transmission filename suffix after the hazy image stem (default: _tabpfn.png).",
    )
    parser.add_argument("--dark-patch-size", type=int, default=15)
    parser.add_argument("--top-fraction", type=float, default=0.001)
    parser.add_argument("--t-min", type=float, default=0.1)
    args = parser.parse_args()
    if args.dark_patch_size < 1 or args.dark_patch_size % 2 != 1:
        parser.error("--dark-patch-size must be a positive odd integer")
    if not 0 < args.top_fraction <= 1 or not 0 < args.t_min <= 1:
        parser.error("--top-fraction and --t-min must be in (0, 1]")
    if not args.hazy_dir.is_dir() or not args.trans_dir.is_dir():
        parser.error("Hazy and transmission directories must exist")
    return args


def main() -> None:
    args = parse_args()
    pairs = [
        (path, args.trans_dir / f"{path.stem}{args.trans_suffix}")
        for path in sorted(args.hazy_dir.iterdir())
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        and (args.trans_dir / f"{path.stem}{args.trans_suffix}").is_file()
    ]
    if not pairs:
        raise ValueError("No hazy images have matching refined transmission maps.")
    if len({path.stem for path, _ in pairs}) != len(pairs):
        raise ValueError("Hazy image stems must be unique to avoid output collisions.")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for hazy_path, trans_path in pairs:
        hazy = load_rgb(hazy_path)
        transmission = load_transmission(trans_path, hazy.shape[:2])
        atmosphere = estimate_atmospheric_light(hazy, args.dark_patch_size, args.top_fraction)
        recovered = recover_radiance(hazy, transmission, atmosphere, args.t_min)
        output_path = args.output_dir / f"{hazy_path.stem}_dehazed.png"
        output = cv2.cvtColor((recovered * 255.0).round().astype(np.uint8), cv2.COLOR_RGB2BGR)
        if not cv2.imwrite(str(output_path), output):
            raise OSError(f"Could not write {output_path}")
        records.append({
            "image": hazy_path.name, "transmission": str(trans_path),
            "A_R": float(atmosphere[0]), "A_G": float(atmosphere[1]),
            "A_B": float(atmosphere[2]), "dark_patch_size": args.dark_patch_size,
            "top_fraction": args.top_fraction, "t_min": args.t_min,
        })
        print(f"Saved {output_path}; atmospheric light RGB: {atmosphere.round(4).tolist()}")
    with (args.output_dir / "atmospheric_light.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    print(f"Dehazed {len(pairs)} images using matching refined transmission maps.")


if __name__ == "__main__":
    main()
