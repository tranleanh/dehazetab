#!/usr/bin/env python3
"""Predict patch transmission with TabPFN and refine maps with guided filtering."""

from __future__ import annotations

import argparse
import os
from collections.abc import Iterator
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from scipy import ndimage


IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
IMAGE_SIZE = (256, 256)
PATCH_SIZE = 16
STRIDE = 8
PATCHES_PER_IMAGE = 100
MIN_CENTER_DISTANCE = 12.0
TRAIN_PAIR_COUNT = 20
TEST_PAIR_COUNT = 2
EPSILON = 1e-6


def configure_writable_caches(cache_dir: Path) -> None:
    # The 2,000-row training table also needs to work when CUDA is unavailable.
    os.environ.setdefault("TABPFN_ALLOW_CPU_LARGE_DATASET", "1")
    cache_dir = cache_dir.resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ["SKB_DATA_DIRECTORY"] = str(cache_dir / "skrub")
    os.environ["MPLCONFIGDIR"] = str(cache_dir / "matplotlib")
    os.environ["HF_HOME"] = str(cache_dir / "huggingface")
    os.environ["TABPFN_MODEL_CACHE_DIR"] = str(cache_dir / "tabpfn")
    for key in ("SKB_DATA_DIRECTORY", "MPLCONFIGDIR", "HF_HOME", "TABPFN_MODEL_CACHE_DIR"):
        Path(os.environ[key]).mkdir(parents=True, exist_ok=True)


def load_rgb(path: Path, resize: bool = True) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Could not read image: {path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    if resize:
        image = cv2.resize(image, IMAGE_SIZE, interpolation=cv2.INTER_AREA)
    return image.astype(np.float32) / 255.0


def test_images_from_split(hazy_dir: Path) -> list[Path]:
    hazy_images = sorted(
        path
        for path in hazy_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    test_images = hazy_images[TRAIN_PAIR_COUNT : TRAIN_PAIR_COUNT + TEST_PAIR_COUNT]
    if len(test_images) != TEST_PAIR_COUNT:
        raise ValueError(f"Expected {TEST_PAIR_COUNT} test images, found {len(test_images)}.")
    return test_images


def patch_positions() -> Iterator[tuple[int, int]]:
    for y in range(0, IMAGE_SIZE[1] - PATCH_SIZE + 1, STRIDE):
        for x in range(0, IMAGE_SIZE[0] - PATCH_SIZE + 1, STRIDE):
            yield y, x


def entropy_0_to_1(gray_patch: np.ndarray) -> float:
    hist = np.histogram(gray_patch, bins=16, range=(0.0, 1.0))[0].astype(np.float32)
    probabilities = hist / max(float(hist.sum()), EPSILON)
    probabilities = probabilities[probabilities > 0.0]
    entropy = -np.sum(probabilities * np.log2(probabilities))
    return float(entropy / np.log2(16.0))


def patch_score(patch: np.ndarray) -> float:
    gray = 0.299 * patch[:, :, 0] + 0.587 * patch[:, :, 1] + 0.114 * patch[:, :, 2]
    dark = patch.min(axis=2)
    hsv = cv2.cvtColor((patch * 255.0).astype(np.uint8), cv2.COLOR_RGB2HSV).astype(np.float32)
    saturation = hsv[:, :, 1] / 255.0

    grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    gradient = np.sqrt((grad_x * grad_x) + (grad_y * grad_y))

    return float(
        (0.25 * gray.std())
        + (0.20 * (gray.max() - gray.min()))
        + (0.20 * gradient.mean())
        + (0.15 * entropy_0_to_1(gray))
        + (0.10 * dark.std())
        + (0.10 * saturation.std())
    )


def selected_patch_positions(image: np.ndarray) -> list[tuple[int, int]]:
    candidates: list[tuple[float, int, int]] = []
    for y, x in patch_positions():
        score = patch_score(image[y : y + PATCH_SIZE, x : x + PATCH_SIZE, :])
        candidates.append((score, y, x))

    candidates.sort(reverse=True)
    selected: list[tuple[int, int]] = []
    selected_centers: list[tuple[float, float]] = []

    for _, y, x in candidates:
        center = (y + PATCH_SIZE / 2.0, x + PATCH_SIZE / 2.0)
        if all(
            np.hypot(center[0] - other[0], center[1] - other[1]) >= MIN_CENTER_DISTANCE
            for other in selected_centers
        ):
            selected.append((y, x))
            selected_centers.append(center)
            if len(selected) == PATCHES_PER_IMAGE:
                break

    if len(selected) < PATCHES_PER_IMAGE:
        already_selected = set(selected)
        for _, y, x in candidates:
            if (y, x) in already_selected:
                continue
            selected.append((y, x))
            if len(selected) == PATCHES_PER_IMAGE:
                break

    return selected


def read_table(workbook_path: Path) -> tuple[pd.DataFrame, np.ndarray, pd.DataFrame]:
    train_df = pd.read_excel(workbook_path, sheet_name="train")
    test_df = pd.read_excel(workbook_path, sheet_name="test")
    if "transmission" not in train_df.columns or "transmission" not in test_df.columns:
        raise ValueError("Workbook must contain a 'transmission' column.")

    feature_columns = [col for col in train_df.columns if col != "transmission"]
    X_train = train_df[feature_columns].astype(np.float32)
    y_train = train_df["transmission"].to_numpy(dtype=np.float32)
    X_test = test_df[feature_columns].astype(np.float32)
    return X_train, y_train, X_test


def fit_predict_tabpfn(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_test: pd.DataFrame,
    device: str,
    n_estimators: int,
    version: str,
    model_path: Path | None,
    save_fit: Path | None = None,
) -> np.ndarray:
    from tabpfn import TabPFNRegressor  # Imported after writable caches are configured.
    from tabpfn.constants import ModelVersion

    common_kwargs = dict(
        n_estimators=n_estimators,
        device=device,
        fit_mode="low_memory",
        memory_saving_mode=True,
        keep_cache_on_device=False,
        inference_precision="auto",
        random_state=42,
        show_progress_bar=True,
    )

    if model_path is not None:
        regressor = TabPFNRegressor(model_path=str(model_path), **common_kwargs)
    else:
        version_map = {
            "v2": ModelVersion.V2,
            "v2.5": ModelVersion.V2_5,
            "v2.6": ModelVersion.V2_6,
            "v3": ModelVersion.V3,
            "v3.5-fast": ModelVersion.V3_5_FAST,
            "v3.5": ModelVersion.V3_5,
        }
        regressor = TabPFNRegressor.create_default_for_version(
            version_map[version],
            **common_kwargs,
        )

    regressor.fit(X_train, y_train)
    if save_fit is not None:
        regressor.save_fit_state(save_fit)
        print(f"Saved fitted TabPFN state to {save_fit}")
    predictions = regressor.predict(X_test)
    return np.asarray(predictions, dtype=np.float32)


def predict_from_saved_fit(fit_path: Path, X_test: pd.DataFrame, device: str) -> np.ndarray:
    from tabpfn import TabPFNRegressor

    regressor = TabPFNRegressor.load_from_fit_state(fit_path, device=device)
    expected_columns = list(regressor.feature_names_in_)
    if set(X_test.columns) != set(expected_columns) or not X_test.columns.is_unique:
        raise ValueError("Test feature columns do not match the saved regressor.")
    predictions = regressor.predict(X_test[expected_columns].astype(np.float32))
    return np.asarray(predictions, dtype=np.float32)


def fill_missing_nearest(map_256: np.ndarray, known_mask: np.ndarray) -> np.ndarray:
    if known_mask.all():
        return map_256
    if not known_mask.any():
        raise ValueError("No known pixels are available for reconstruction.")
    _, indices = ndimage.distance_transform_edt(~known_mask, return_indices=True)
    filled = map_256.copy()
    filled[~known_mask] = map_256[indices[0][~known_mask], indices[1][~known_mask]]
    return filled


def reconstruct_map(predictions: np.ndarray, positions: list[tuple[int, int]]) -> np.ndarray:
    accum = np.zeros(IMAGE_SIZE[::-1], dtype=np.float32)
    counts = np.zeros(IMAGE_SIZE[::-1], dtype=np.float32)

    for prediction, (y, x) in zip(predictions, positions, strict=True):
        y_slice = slice(y, y + PATCH_SIZE)
        x_slice = slice(x, x + PATCH_SIZE)
        accum[y_slice, x_slice] += float(prediction)
        counts[y_slice, x_slice] += 1.0

    known_mask = counts > 0.0
    map_256 = np.zeros_like(accum)
    map_256[known_mask] = accum[known_mask] / counts[known_mask]
    map_256 = fill_missing_nearest(map_256, known_mask)
    return np.clip(map_256, 0.0, 1.0)


def guided_filter(
    hazy_rgb: np.ndarray,
    transmission: np.ndarray,
    radius: int = 16,
    eps: float = 1e-3,
) -> np.ndarray:
    """Filter transmission using normalized hazy luminance as guidance.

    Implements the scalar guided filter's local linear model using box means.
    Radius is measured at 256x256; eps regularizes variance in [0, 1] intensity
    units. Larger radius/eps generally give stronger smoothing.
    """
    if radius < 1 or not np.isfinite(eps) or eps <= 0:
        raise ValueError("Guided radius must be >= 1 and eps must be finite and > 0.")
    if hazy_rgb.shape != (*transmission.shape, 3) or transmission.ndim != 2:
        raise ValueError("Expected aligned RGB guidance and a 2D transmission map.")
    if not np.isfinite(hazy_rgb).all() or not np.isfinite(transmission).all():
        raise ValueError("Guidance and transmission must contain only finite values.")
    guide = cv2.cvtColor(hazy_rgb.astype(np.float32), cv2.COLOR_RGB2GRAY).astype(np.float64)
    source = transmission.astype(np.float64)
    window = (2 * radius + 1, 2 * radius + 1)

    def mean(values: np.ndarray) -> np.ndarray:
        return cv2.boxFilter(
            values, -1, window, normalize=True, borderType=cv2.BORDER_REFLECT
        )

    mean_guide = mean(guide)
    mean_source = mean(source)
    variance = np.maximum(mean(guide * guide) - mean_guide * mean_guide, 0.0)
    covariance = mean(guide * source) - mean_guide * mean_source
    a = covariance / (variance + eps)
    b = mean_source - a * mean_guide
    refined = mean(a) * guide + mean(b)
    return np.clip(refined, 0.0, 1.0).astype(np.float32)


def save_outputs(
    predictions: np.ndarray,
    test_images: list[Path],
    output_dir: Path,
    guided_radius: int = 16,
    guided_eps: float = 1e-3,
) -> None:
    expected_predictions = len(test_images) * PATCHES_PER_IMAGE
    if len(predictions) != expected_predictions:
        raise ValueError(
            f"Expected {expected_predictions} predictions, got {len(predictions)}."
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []

    for image_idx, hazy_path in enumerate(test_images):
        resized_hazy = load_rgb(hazy_path, resize=True)
        original = cv2.imread(str(hazy_path), cv2.IMREAD_COLOR)
        if original is None:
            raise ValueError(f"Could not read image: {hazy_path}")
        original_height, original_width = original.shape[:2]

        start = image_idx * PATCHES_PER_IMAGE
        end = start + PATCHES_PER_IMAGE
        image_predictions = np.clip(predictions[start:end], 0.0, 1.0)
        positions = selected_patch_positions(resized_hazy)

        for patch_idx, ((y, x), prediction) in enumerate(
            zip(positions, image_predictions, strict=True)
        ):
            rows.append(
                {
                    "image": hazy_path.stem,
                    "patch_index": patch_idx,
                    "y": y,
                    "x": x,
                    "prediction": float(prediction),
                }
            )

        unrefined = reconstruct_map(image_predictions, positions)
        map_256 = guided_filter(resized_hazy, unrefined, guided_radius, guided_eps)
        map_original = cv2.resize(
            map_256,
            (original_width, original_height),
            interpolation=cv2.INTER_LINEAR,
        )

        cv2.imwrite(
            str(output_dir / f"{hazy_path.stem}_tabpfn_unrefined_256.png"),
            (unrefined * 255.0).round().astype(np.uint8),
        )
        cv2.imwrite(
            str(output_dir / f"{hazy_path.stem}_tabpfn_256.png"),
            (map_256 * 255.0).round().astype(np.uint8),
        )
        cv2.imwrite(
            str(output_dir / f"{hazy_path.stem}_tabpfn.png"),
            (map_original * 255.0).round().astype(np.uint8),
        )

    pd.DataFrame(rows).to_csv(output_dir / "tabpfn_patch_predictions.csv", index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Use lightweight TabPFN regression to reconstruct test transmission maps."
    )
    parser.add_argument(
        "--workbook",
        type=Path,
        default=Path("testdata/patch_table_method02_handcrafted_features_v2.xlsx"),
    )
    parser.add_argument("--hazy-dir", type=Path, default=Path("testdata/hazy"))
    parser.add_argument("--output-dir", type=Path, default=Path("testdata/tabpfn_reconstructed"))
    parser.add_argument("--cache-dir", type=Path, default=Path(".cache"))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--n-estimators", type=int, default=1)
    parser.add_argument(
        "--save-fit", type=Path, default=Path("models/tabpfn_transmission.tabpfn_fit"),
        help="Save fitted training context; pretrained weights remain in the model cache.",
    )
    parser.add_argument(
        "--load-fit", type=Path,
        help="Reuse a saved .tabpfn_fit instead of fitting; only the test sheet is read.",
    )
    parser.add_argument(
        "--guided-radius", type=int, default=16,
        help="Guided filter radius in pixels at 256x256 (default: 16).",
    )
    parser.add_argument(
        "--guided-eps", type=float, default=1e-3,
        help="Guided filter regularization for [0,1] guidance (default: 0.001).",
    )
    parser.add_argument(
        "--version",
        choices=["v2", "v2.5", "v2.6", "v3", "v3.5-fast", "v3.5"],
        default="v2",
        help="TabPFN model version. v2 is the lightweight ungated default.",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=None,
        help="Optional local TabPFN checkpoint path. Overrides --version.",
    )
    args = parser.parse_args()
    if args.save_fit.suffix != ".tabpfn_fit":
        parser.error("--save-fit must end with .tabpfn_fit")
    if args.load_fit is not None and not args.load_fit.is_file():
        parser.error("--load-fit must refer to an existing fitted state file")
    if args.guided_radius < 1 or not np.isfinite(args.guided_eps) or args.guided_eps <= 0:
        parser.error("--guided-radius must be >= 1 and --guided-eps must be finite and > 0")
    return args


def main() -> None:
    args = parse_args()
    configure_writable_caches(args.cache_dir)

    test_images = test_images_from_split(args.hazy_dir)
    if args.load_fit is not None:
        test_df = pd.read_excel(args.workbook, sheet_name="test")
        X_test = test_df.drop(columns="transmission", errors="ignore")
        predictions = predict_from_saved_fit(args.load_fit, X_test, args.device)
        print(f"Loaded fitted TabPFN state from {args.load_fit}")
    else:
        X_train, y_train, X_test = read_table(args.workbook)
        predictions = fit_predict_tabpfn(
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            device=args.device,
            n_estimators=args.n_estimators,
            version=args.version,
            model_path=args.model_path,
            save_fit=args.save_fit,
        )
        print(f"Training rows: {len(X_train)}")
    save_outputs(predictions, test_images, args.output_dir, args.guided_radius, args.guided_eps)

    print(f"Predicted patches: {len(predictions)}")
    print(f"Test images: {[path.stem for path in test_images]}")
    print(f"Prediction range: {float(predictions.min()):.4f} to {float(predictions.max()):.4f}")
    print(f"Guided refinement: radius={args.guided_radius}, eps={args.guided_eps}")
    print(f"Saved outputs to {args.output_dir}")


if __name__ == "__main__":
    main()
