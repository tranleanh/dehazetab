#!/usr/bin/env python3
"""Build a ResNet-50 image index, or retrieve top-k matches for --query IMAGE.

Defaults to testdata/hazy and testdata/embeddings/resnet50.npz. Uses ImageNet
pretrained global-average-pool features, not classifier logits. Both indexed
and query images use the weights' resize/center-crop/normalization pipeline.
L2-normalized features are ranked by cosine similarity (higher is closer).
No training or clean images are required.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps
import torch
from torch import nn
import torch.nn.functional as F
from torchvision.models import ResNet50_Weights, resnet50

from .utils import IMAGE_EXTENSIONS


WEIGHTS = ResNet50_Weights.IMAGENET1K_V2
MODEL_ID = "resnet50.IMAGENET1K_V2.avgpool"
DIMENSION = 2048


def load_encoder(device: str, cache_dir: Path) -> tuple[nn.Module, object]:
    torch.hub.set_dir(str(cache_dir.resolve()))
    model = resnet50(weights=WEIGHTS)
    model.fc = nn.Identity()
    return model.to(device).eval(), WEIGHTS.transforms()


@torch.inference_mode()
def encode_images(
    paths: list[Path], model: nn.Module, transform, device: str, batch_size: int,
) -> np.ndarray:
    if not paths:
        raise ValueError("No images to encode.")
    batches = []
    for start in range(0, len(paths), batch_size):
        tensors = []
        for path in paths[start:start + batch_size]:
            with Image.open(path) as image:
                tensors.append(transform(ImageOps.exif_transpose(image).convert("RGB")))
        raw = model(torch.stack(tensors).to(device)).float()
        if not torch.isfinite(raw).all() or (raw.norm(dim=1) == 0).any():
            raise ValueError("Encoder returned invalid or zero feature vectors.")
        batches.append(F.normalize(raw, p=2, dim=1).cpu().numpy())
    return np.concatenate(batches).astype(np.float32)


def build_index(args: argparse.Namespace, model: nn.Module, transform) -> None:
    paths = sorted(
        path for path in args.hazy_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    embeddings = encode_images(paths, model, transform, args.device, args.batch_size)
    args.index.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "format_version": 1, "model": MODEL_ID, "dimension": DIMENSION,
        "normalization": "l2", "similarity": "cosine",
        "preprocessing": str(transform), "exif_transpose": True,
        "image_root": os.path.relpath(args.hazy_dir.resolve(), args.index.parent.resolve()),
        "torch_version": torch.__version__,
    }
    np.savez_compressed(
        args.index, embeddings=embeddings,
        filenames=np.asarray([path.name for path in paths]),
        metadata=np.asarray(json.dumps(metadata)),
    )
    print(f"Saved {len(paths)} embeddings of dimension {DIMENSION} to {args.index}")


def retrieve(
    query: Path, index_path: Path, model: nn.Module, transform, device: str, top_k: int = 5,
) -> list[tuple[Path, float]]:
    with np.load(index_path, allow_pickle=False) as index:
        embeddings = index["embeddings"]
        filenames = index["filenames"]
        metadata = json.loads(str(index["metadata"].item()))
    if (
        metadata.get("format_version") != 1 or metadata.get("model") != MODEL_ID
        or metadata.get("preprocessing") != str(transform)
        or metadata.get("normalization") != "l2"
        or metadata.get("exif_transpose") is not True
    ):
        raise ValueError("Index encoder/preprocessing differs from this script; rebuild the index.")
    if (
        filenames.ndim != 1 or embeddings.shape != (len(filenames), DIMENSION)
        or not len(filenames) or not np.isfinite(embeddings).all()
        or not np.allclose(np.linalg.norm(embeddings, axis=1), 1, atol=1e-5)
    ):
        raise ValueError("Invalid embedding index.")
    vector = encode_images([query], model, transform, device, 1)[0]
    similarities = np.clip(embeddings @ vector, -1.0, 1.0)
    order = np.argsort(-similarities, kind="stable")[:top_k]
    root = (index_path.parent / metadata["image_root"]).resolve()
    return [(root / str(filenames[i]), float(similarities[i])) for i in order]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hazy-dir", type=Path, default=Path("testdata/hazy"))
    parser.add_argument("--index", type=Path, default=Path("testdata/embeddings/resnet50.npz"))
    parser.add_argument("--query", type=Path, help="Retrieve matches instead of rebuilding the index.")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--cache-dir", type=Path, default=Path(".cache/torch"))
    args = parser.parse_args()
    if min(args.top_k, args.batch_size, args.threads) < 1:
        parser.error("--top-k, --batch-size and --threads must be positive")
    if args.index.suffix != ".npz":
        parser.error("--index must end with .npz")
    if args.query is not None:
        if not args.query.is_file() or not args.index.is_file():
            parser.error("Query image and saved index must exist; first run without --query")
    elif not args.hazy_dir.is_dir():
        parser.error("--hazy-dir must be an existing image directory")
    if args.device == "auto":
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is unavailable; use --device cpu")
    return args


def main() -> None:
    args = parse_args()
    torch.set_num_threads(args.threads)
    model, transform = load_encoder(args.device, args.cache_dir)
    print(f"Encoder: {MODEL_ID}; device: {args.device}")
    if args.query is None:
        build_index(args, model, transform)
    else:
        matches = retrieve(args.query, args.index, model, transform, args.device, args.top_k)
        for rank, (path, score) in enumerate(matches, 1):
            print(f"{rank}. {path}  cosine_similarity={score:.6f}")


if __name__ == "__main__":
    main()
