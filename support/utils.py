"""Shared image loading, patch selection, handcrafted features and normalization."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import cv2
import numpy as np


IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
IMAGE_SIZE = (256, 256)
PATCH_SIZE = 16
STRIDE = 8
PATCHES_PER_IMAGE = 100
MIN_CENTER_DISTANCE = 12.0
EXPECTED_PAIR_COUNT = 22
TRAIN_PAIR_COUNT = 20
TEST_PAIR_COUNT = 2
EPSILON = 1e-6


FEATURE_COLUMNS = [
    "rgb_r_mean",
    "rgb_r_std",
    "rgb_r_min",
    "rgb_r_max",
    "rgb_g_mean",
    "rgb_g_std",
    "rgb_g_min",
    "rgb_g_max",
    "rgb_b_mean",
    "rgb_b_std",
    "rgb_b_min",
    "rgb_b_max",
    "gray_mean",
    "gray_std",
    "gray_min",
    "gray_max",
    "dark_min",
    "dark_mean",
    "dark_std",
    "dark_p10",
    "dark_p25",
    "dark_p50",
    "dark_p75",
    "dark_p90",
    "hsv_s_mean",
    "hsv_s_std",
    "hsv_v_mean",
    "hsv_v_std",
    "intensity_range",
    "rms_contrast",
    "gradient_mean",
    "gradient_std",
    "edge_density",
    "entropy",
    "laplacian_variance",
    "mean_r_minus_g",
    "mean_g_minus_b",
    "mean_r_minus_b",
    "ratio_r_over_g_mean",
    "ratio_r_over_g_std",
    "ratio_g_over_b_mean",
    "ratio_g_over_b_std",
    "ratio_r_over_b_mean",
    "ratio_r_over_b_std",
]


def load_rgb(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Could not read image: {path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, IMAGE_SIZE, interpolation=cv2.INTER_AREA)
    return image.astype(np.float32) / 255.0


def load_transmission(path: Path) -> np.ndarray:
    transmission = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if transmission is None:
        raise ValueError(f"Could not read transmission map: {path}")
    transmission = cv2.resize(transmission, IMAGE_SIZE, interpolation=cv2.INTER_AREA)
    return transmission.astype(np.float32) / 255.0


def paired_images(hazy_dir: Path, trans_dir: Path) -> list[tuple[Path, Path]]:
    trans_by_stem = {
        path.stem: path
        for path in trans_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    }

    pairs: list[tuple[Path, Path]] = []
    for hazy_path in sorted(hazy_dir.iterdir()):
        if not hazy_path.is_file() or hazy_path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        trans_path = trans_by_stem.get(hazy_path.stem)
        if trans_path is None:
            print(f"Skipping {hazy_path.name}: no transmission map with matching stem")
            continue
        pairs.append((hazy_path, trans_path))
    return pairs


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


def clipped_ratio_stats(numerator: np.ndarray, denominator: np.ndarray) -> tuple[float, float]:
    ratio = numerator / np.maximum(denominator, EPSILON)
    ratio = np.clip(ratio, 0.0, 4.0) / 4.0
    return float(ratio.mean()), float(ratio.std())


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


def selected_patch_positions(
    image: np.ndarray, patches_per_image: int = PATCHES_PER_IMAGE,
) -> list[tuple[int, int]]:
    if not 1 <= patches_per_image <= len(list(patch_positions())):
        raise ValueError("patches_per_image must be between 1 and the number of candidate patches.")
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
            if len(selected) == patches_per_image:
                break

    if len(selected) < patches_per_image:
        already_selected = set(selected)
        for _, y, x in candidates:
            if (y, x) in already_selected:
                continue
            selected.append((y, x))
            if len(selected) == patches_per_image:
                break

    return selected


def extract_handcrafted_features(patch: np.ndarray) -> np.ndarray:
    r = patch[:, :, 0]
    g = patch[:, :, 1]
    b = patch[:, :, 2]

    gray = 0.299 * r + 0.587 * g + 0.114 * b
    dark = patch.min(axis=2)
    hsv = cv2.cvtColor((patch * 255.0).astype(np.uint8), cv2.COLOR_RGB2HSV).astype(np.float32)
    saturation = hsv[:, :, 1] / 255.0
    value = hsv[:, :, 2] / 255.0

    grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    gradient = np.sqrt((grad_x * grad_x) + (grad_y * grad_y))

    gray_u8 = (gray * 255.0).astype(np.uint8)
    edges = cv2.Canny(gray_u8, threshold1=50, threshold2=150)
    laplacian = cv2.Laplacian(gray, cv2.CV_32F, ksize=3)

    r_over_g_mean, r_over_g_std = clipped_ratio_stats(r, g)
    g_over_b_mean, g_over_b_std = clipped_ratio_stats(g, b)
    r_over_b_mean, r_over_b_std = clipped_ratio_stats(r, b)

    features: list[float] = []
    for channel in (r, g, b):
        features.extend(
            [
                float(channel.mean()),
                float(channel.std()),
                float(channel.min()),
                float(channel.max()),
            ]
        )

    features.extend(
        [
            float(gray.mean()),
            float(gray.std()),
            float(gray.min()),
            float(gray.max()),
            float(dark.min()),
            float(dark.mean()),
            float(dark.std()),
            float(np.percentile(dark, 10)),
            float(np.percentile(dark, 25)),
            float(np.percentile(dark, 50)),
            float(np.percentile(dark, 75)),
            float(np.percentile(dark, 90)),
            float(saturation.mean()),
            float(saturation.std()),
            float(value.mean()),
            float(value.std()),
            float(gray.max() - gray.min()),
            float(gray.std()),
            float(gradient.mean()),
            float(gradient.std()),
            float(np.count_nonzero(edges) / edges.size),
            entropy_0_to_1(gray),
            float(laplacian.var()),
            float((r - g).mean()),
            float((g - b).mean()),
            float((r - b).mean()),
            r_over_g_mean,
            r_over_g_std,
            g_over_b_mean,
            g_over_b_std,
            r_over_b_mean,
            r_over_b_std,
        ]
    )

    return np.asarray(features, dtype=np.float32)


def clustered_patch_positions(
    features: np.ndarray, random_state: int = 42, *, patches_per_image: int = PATCHES_PER_IMAGE,
) -> list[tuple[int, int]]:
    """Select one member nearest each K-means centroid in the supplied feature space.

    Duplicate embeddings can produce empty clusters. Fill those slots with
    unused patches nearest the corresponding centroid to keep the requested count.
    """
    from sklearn.cluster import KMeans

    positions = list(patch_positions())
    if not 1 <= patches_per_image <= len(positions):
        raise ValueError("patches_per_image must be between 1 and the number of candidate patches.")
    values = np.asarray(features, dtype=np.float32)
    if values.ndim != 2 or values.shape[0] != len(positions) or values.shape[1] == 0:
        raise ValueError("Expected one feature vector per candidate patch in row-major order.")
    if not np.isfinite(values).all():
        raise ValueError("Patch features must be finite.")
    model = KMeans(n_clusters=patches_per_image, random_state=random_state, n_init=10)
    labels = model.fit_predict(values)
    selected = []
    empty = []
    for cluster, center in enumerate(model.cluster_centers_):
        members = np.flatnonzero(labels == cluster)
        if not len(members):
            empty.append(cluster)
            continue
        distances = np.sum((values[members] - center) ** 2, axis=1)
        selected.append(int(members[np.argmin(distances)]))
    for cluster in empty:
        remaining = np.setdiff1d(np.arange(len(values)), selected)
        distances = np.sum((values[remaining] - model.cluster_centers_[cluster]) ** 2, axis=1)
        selected.append(int(remaining[np.argmin(distances)]))
    return [positions[i] for i in selected]


def iter_patch_features(
    pairs: list[tuple[Path, Path]],
    include_targets: bool,
    positions_by_path: dict[Path, list[tuple[int, int]]] | None = None,
) -> Iterator[tuple[np.ndarray, float | None]]:
    for hazy_path, trans_path in pairs:
        hazy = load_rgb(hazy_path)
        transmission = load_transmission(trans_path)

        positions = selected_patch_positions(hazy) if positions_by_path is None else positions_by_path[hazy_path]
        for y, x in positions:
            y_slice = slice(y, y + PATCH_SIZE)
            x_slice = slice(x, x + PATCH_SIZE)
            features = extract_handcrafted_features(hazy[y_slice, x_slice, :])
            if include_targets:
                target = float(transmission[y_slice, x_slice].mean())
            else:
                target = None
            yield features, target


def training_standardization_stats(
    train_pairs: list[tuple[Path, Path]],
    positions_by_path: dict[Path, list[tuple[int, int]]] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    count = 0
    mean = np.zeros(len(FEATURE_COLUMNS), dtype=np.float64)
    m2 = np.zeros(len(FEATURE_COLUMNS), dtype=np.float64)

    for features, _ in iter_patch_features(train_pairs, include_targets=True, positions_by_path=positions_by_path):
        count += 1
        delta = features - mean
        mean += delta / count
        delta2 = features - mean
        m2 += delta * delta2

    if count < 2:
        raise ValueError("At least two training patches are required for standardization.")

    std = np.sqrt(m2 / (count - 1))
    std[std < EPSILON] = 1.0
    return mean.astype(np.float32), std.astype(np.float32)


def standardized_row(
    features: np.ndarray,
    target: float | None,
    mean: np.ndarray,
    std: np.ndarray,
) -> list[float | None]:
    standardized = ((features - mean) / std).astype(np.float32)
    row = standardized.tolist()
    row.append(target)
    return row
