#!/usr/bin/env python3
"""Batch dehaze a folder of unseen hazy images.

For each image in --input-dir:
1. Retrieve N+1 candidates, exclude the query itself, and keep N references
   (--num-references, default 5).
2. Load N * P labeled context rows, where P is --patches-per-image or the
   count saved in normalization.npz (legacy default: 100).
3. Extract standardized handcrafted features for the query image patches.
4. Predict patch transmission with pretrained TabPFN v2.
5. Reconstruct and refine the transmission map.
6. Recover and save ONLY the final dehazed image.

No intermediate CSV, transmission map, NPZ, or timing JSON is saved.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import time
import torch
import numpy as np
import pandas as pd

from support import utils, dehaze
import support.similarity as retrieval
import support.construct as reconstruction


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def load_normalization(table_dir: Path):
    """Load and validate the training normalization statistics once."""
    path = table_dir / "normalization.npz"
    with np.load(path, allow_pickle=False) as stats:
        mean = stats["mean"]
        std = stats["std"]
        names = stats["feature_columns"].tolist()
        training_stems = set(stats["image_stems"].tolist())

    if (
        names != utils.FEATURE_COLUMNS
        or mean.shape != (len(names),)
        or std.shape != mean.shape
        or not np.isfinite(mean).all()
        or not np.isfinite(std).all()
        or np.any(std <= 0)
    ):
        raise ValueError("Invalid or incompatible training normalization file.")

    return mean, std, training_stems


def resolve_patch_count(table_dir: Path, requested: int | None = None) -> int:
    """Resolve table row count without changing the shared normalization API."""
    with np.load(table_dir / "normalization.npz", allow_pickle=False) as stats:
        stored = stats["patches_per_image"].item() if "patches_per_image" in stats else None
    maximum = len(list(utils.patch_positions()))
    for value in (stored, requested):
        if value is not None and (not isinstance(value, (int, np.integer)) or not 1 <= value <= maximum):
            raise ValueError(f"patches_per_image must be an integer between 1 and {maximum}.")
    if stored is not None and requested is not None and stored != requested:
        raise ValueError(f"Requested {requested} patches per image, but normalization.npz specifies {stored}.")
    return int(requested if requested is not None else stored if stored is not None else utils.PATCHES_PER_IMAGE)


def build_query_data(
    query: Path,
    index: Path,
    table_dir: Path,
    device: str,
    encoder,
    transform,
    mean: np.ndarray,
    std: np.ndarray,
    training_stems: set[str],
    num_references: int = 5,
    patches_per_image: int = utils.PATCHES_PER_IMAGE,
) -> tuple[pd.DataFrame, np.ndarray, pd.DataFrame]:
    """Retrieve context images and build TabPFN context/query features."""
    # Retrieve one extra candidate because the query image itself may already
    # exist in the embedding index. Remove self-matches, then keep the best N.
    if num_references < 1:
        raise ValueError("num_references must be positive.")
    if not 1 <= patches_per_image <= len(list(utils.patch_positions())):
        raise ValueError("patches_per_image must be between 1 and 961.")
    candidates = retrieval.retrieve(
        query, index, encoder, transform, device, top_k=num_references + 1
    )

    matches = []
    seen_stems = set()
    for image_path, score in candidates:
        # Treat the same filename stem as the same image even when the query and
        # indexed copy live in different folders or use different extensions.
        if image_path.stem == query.stem:
            print(
                f"    Skipping self-match: {image_path.name} "
                f"(cosine={score:.6f})",
                flush=True,
            )
            continue
        if image_path.stem in seen_stems:
            continue
        matches.append((image_path, score))
        seen_stems.add(image_path.stem)
        if len(matches) == num_references:
            break

    if len(matches) != num_references:
        raise ValueError(
            f"Requested {num_references} references, but found only {len(matches)} "
            f"distinct non-query images among the top-{num_references + 1} candidates."
        )

    columns = utils.FEATURE_COLUMNS + ["transmission"]
    context_frames: list[pd.DataFrame] = []

    for rank, (image_path, score) in enumerate(matches, 1):
        if image_path.stem not in training_stems:
            raise ValueError(
                f"Image {image_path.name} is not covered by normalization.npz."
            )

        table_path = table_dir / f"{image_path.stem}.csv"
        frame = pd.read_csv(table_path)

        if list(frame.columns) != columns or len(frame) != patches_per_image:
            raise ValueError(
                f"Expected {patches_per_image} rows with the handcrafted "
                f"schema in {table_path}"
            )
        if not np.isfinite(frame.to_numpy(dtype=np.float64)).all():
            raise ValueError(f"Missing or invalid values in {table_path}")
        if not frame["transmission"].between(0, 1).all():
            raise ValueError(f"Transmission targets outside [0,1] in {table_path}")

        context_frames.append(frame)
        print(f"    {rank}. {image_path.name}: cosine={score:.6f}", flush=True)

    image_256 = utils.load_rgb(query)
    query_rows = []

    for y, x in utils.patch_positions():
        patch = image_256[
            y : y + utils.PATCH_SIZE,
            x : x + utils.PATCH_SIZE,
        ]
        features = utils.extract_handcrafted_features(patch)
        query_rows.append(utils.standardized_row(features, None, mean, std))

    query_frame = pd.DataFrame(query_rows, columns=columns)
    if not np.isfinite(
        query_frame[utils.FEATURE_COLUMNS].to_numpy(dtype=np.float32)
    ).all():
        raise ValueError("Query features contain invalid values.")

    context = pd.concat(context_frames, ignore_index=True)
    X_context = context[utils.FEATURE_COLUMNS].astype(np.float32)
    y_context = context["transmission"].to_numpy(dtype=np.float32)
    X_query = query_frame[utils.FEATURE_COLUMNS].astype(np.float32)

    return X_context, y_context, X_query


def save_rgb_png(path: Path, image: np.ndarray) -> None:
    """Save an RGB float image in [0,1] as PNG."""
    pixels = (np.clip(image, 0, 1) * 255).round().astype(np.uint8)
    pixels = cv2.cvtColor(pixels, cv2.COLOR_RGB2BGR)
    if not cv2.imwrite(str(path), pixels):
        raise OSError(f"Could not write {path}")


def process_one(
    query: Path,
    args: argparse.Namespace,
    device: str,
    encoder,
    transform,
    mean: np.ndarray,
    std: np.ndarray,
    training_stems: set[str],
    TabPFNRegressor,
    ModelVersion,
) -> Path:
    """Dehaze one image and return the saved output path."""
    print(f"\nProcessing: {query.name}", flush=True)

    X_context, y_context, X_query = build_query_data(
        query=query,
        index=args.index,
        table_dir=args.table_dir,
        device=device,
        encoder=encoder,
        transform=transform,
        mean=mean,
        std=std,
        training_stems=training_stems,
        num_references=args.num_references,
        patches_per_image=args.patches_per_image,
    )

    positions = list(utils.patch_positions())
    expected_context_rows = args.num_references * args.patches_per_image
    if len(X_context) != expected_context_rows:
        raise ValueError(f"Expected {expected_context_rows} context rows, got {len(X_context)}.")
    if len(X_query) != len(positions):
        raise ValueError(f"Expected {len(positions)} query rows, got {len(X_query)}.")

    model = TabPFNRegressor.create_default_for_version(
        ModelVersion.V2,
        n_estimators=1,
        device=device,
        fit_mode="low_memory",
        memory_saving_mode=True,
        keep_cache_on_device=False,
        inference_precision="auto",
        random_state=42,
        show_progress_bar=False,
    )
    model.fit(X_context, y_context)

    predictions = np.concatenate(
        [
            np.asarray(
                model.predict(X_query.iloc[i : i + args.batch_size]),
                dtype=np.float32,
            )
            for i in range(0, len(X_query), args.batch_size)
        ]
    )

    if predictions.shape != (len(positions),) or not np.isfinite(predictions).all():
        raise ValueError("Invalid transmission predictions.")

    patch_transmission = np.clip(predictions, 0, 1)
    coarse = reconstruction.reconstruct_map(patch_transmission, positions)

    guidance = utils.load_rgb(query)
    refined = reconstruction.guided_filter(
        guidance, coarse, args.guided_radius, args.guided_eps
    )

    original = dehaze.load_rgb(query)
    size = (original.shape[1], original.shape[0])
    refined_original = cv2.resize(refined, size, interpolation=cv2.INTER_LINEAR)

    atmosphere = dehaze.estimate_atmospheric_light(original)
    recovered = dehaze.recover_radiance(
        original, refined_original, atmosphere, args.t_min
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / f"{query.stem}_dehazed.png"
    save_rgb_png(output_path, recovered)
    print(f"  Saved: {output_path}", flush=True)

    del model
    if device == "cuda":
        torch.cuda.empty_cache()

    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Folder containing unseen hazy test images.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Folder where only final dehazed PNG images will be saved.",
    )
    parser.add_argument(
        "--index",
        type=Path,
        default=Path(__file__).resolve().parent / "tables/embeddings/resnet50.npz",
        help="Embedding index produced by code03_extract_emb_for_similarity.py.",
    )
    parser.add_argument(
        "--table-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "tables/table_method01_handcrafted",
        help="Directory containing per-image training tables and normalization.npz.",
    )
    parser.add_argument(
        "--device", choices=["auto", "cpu", "cuda"], default="auto"
    )
    parser.add_argument(
        "--retrieval-cache-dir",
        type=Path,
        default=Path(__file__).resolve().parent / ".cache/torch",
        help="Cache used by the image retrieval encoder.",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path(__file__).resolve().parent / ".cache",
        help="Writable cache used by TabPFN/reconstruction.",
    )
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-references", type=int, default=5,
                        help="Number of retrieved context images (default: 5).")
    parser.add_argument("--patches-per-image", type=int, default=None,
                        help="Rows per reference table; inferred from normalization.npz when omitted, otherwise 100 for legacy tables.")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--guided-radius", type=int, default=16)
    parser.add_argument("--guided-eps", type=float, default=0.001)
    parser.add_argument("--t-min", type=float, default=0.1)

    args = parser.parse_args()

    if not args.input_dir.is_dir():
        parser.error("--input-dir must be an existing directory.")
    if args.input_dir.resolve() == args.output_dir.resolve():
        parser.error("Use a separate output directory.")
    if not args.index.is_file():
        parser.error("Embedding index does not exist.")
    if not (args.table_dir / "normalization.npz").is_file():
        parser.error(
            "Missing normalization.npz; run "
            "code02_tab_per_img_method01_handcrafted.py first."
        )
    if min(args.batch_size, args.threads, args.guided_radius, args.num_references) < 1:
        parser.error("Batch size, threads, guided radius, and number of references must be positive.")
    if (
        not np.isfinite(args.guided_eps)
        or args.guided_eps <= 0
        or not 0 < args.t_min <= 1
    ):
        parser.error("Invalid guided epsilon or minimum transmission.")

    try:
        args.patches_per_image = resolve_patch_count(args.table_dir, args.patches_per_image)
    except ValueError as exc:
        parser.error(str(exc))
    return args


def main() -> None:
    args = parse_args()

    reconstruction.configure_writable_caches(args.cache_dir)
    from tabpfn import TabPFNRegressor
    from tabpfn.constants import ModelVersion

    torch.set_num_threads(args.threads)
    cv2.setNumThreads(1)

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA unavailable; use --device cpu.")

    images = sorted(
        p for p in args.input_dir.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not images:
        raise ValueError(f"No supported image files found in {args.input_dir}")
    if len({path.stem for path in images}) != len(images):
        raise ValueError("Input image stems must be unique to avoid output collisions.")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    mean, std, training_stems = load_normalization(args.table_dir)
    with np.load(args.index, allow_pickle=False) as index:
        indexed_stems = {Path(str(name)).stem for name in index["filenames"]}
    missing = indexed_stems - training_stems
    if missing:
        raise ValueError(f"Retrieval index contains images absent from table metadata: {sorted(missing)[:5]}")
    for stem in indexed_stems:
        if not (args.table_dir / f"{stem}.csv").is_file():
            raise ValueError(f"Missing reference table: {stem}.csv")
    if len(indexed_stems) < args.num_references:
        raise ValueError("Not enough indexed reference tables for --num-references.")
    encoder, transform = retrieval.load_encoder(device, args.retrieval_cache_dir)

    print(f"Device: {device}", flush=True)
    print(f"Context: {args.num_references} references x {args.patches_per_image} rows = {args.num_references * args.patches_per_image}", flush=True)
    print(f"Found {len(images)} test images in {args.input_dir}", flush=True)
    print(f"Saving dehazed images to {args.output_dir}", flush=True)

    succeeded = 0
    failed = 0

    for i, query in enumerate(images, 1):
        print(f"\n[{i}/{len(images)}]", flush=True)
        try:
            starttime = time.time()
            process_one(
                query=query,
                args=args,
                device=device,
                encoder=encoder,
                transform=transform,
                mean=mean,
                std=std,
                training_stems=training_stems,
                TabPFNRegressor=TabPFNRegressor,
                ModelVersion=ModelVersion,
            )
            succeeded += 1
            proctime = time.time() - starttime
            print(f'Proctime: {proctime}')
        except Exception as exc:
            failed += 1
            print(f"  ERROR: {query.name}: {exc}", flush=True)

    print(
        f"\nFinished. Success: {succeeded}; Failed: {failed}; "
        f"Output folder: {args.output_dir}",
        flush=True,
    )

    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
