# DehazeTab

**Single-image dehazing with retrieval-guided tabular in-context learning.**

DehazeTab represents image patches as feature rows and uses pretrained TabPFN to estimate their transmission values. Retrieved reference tables provide labeled context for each new image, without gradient-based fine-tuning of TabPFN.

This repository provides a standalone inference demo of the **handcrafted-feature variant**. It accepts a folder of hazy images and saves the recovered haze-free images. Clean reference images and ground-truth transmission maps are not required for the test inputs.

## How It Works

1. **Retrieve references:** encode the input with pretrained ResNet-50 and select similar reference images from the saved embedding index.
2. **Build the context:** load the selected reference CSV tables, containing 44 handcrafted features and a transmission target per row.
3. **Describe the input:** resize the query to 256 x 256 and extract features from 961 overlapping 16 x 16 patches with stride 8. Apply the saved training normalization.
4. **Predict transmission:** use pretrained TabPFN v2 with the retrieved context to predict one transmission value per query patch.
5. **Recover the image:** average overlapping predictions, apply guided filtering, resize the transmission map to the original image dimensions, and invert the haze imaging model using Dark Channel Prior atmospheric-light estimation.

The default context uses **5 reference tables**, typically with **100 rows each**. All 961 query patches are retained for prediction.

## Repository Layout

```text
.
├── dehazetab.py                         # Main inference entry point
├── requirements.txt                    # Dependency versions from the tested environment
├── support/                            # Retrieval, features, reconstruction, and dehazing
├── tables/
│   ├── embeddings/resnet50.npz          # Reference-image retrieval index
│   └── table_method01_handcrafted/
│       ├── normalization.npz           # Training statistics and image metadata
│       └── <reference_stem>.csv         # Labeled patch features per reference image
└── testdata/
    ├── inputs/                         # Hazy test images
    └── outputs/                        # Generated dehazed PNGs
```

## Setup

For the existing environment:

```bash
conda activate tfmdehaze
```

For a new environment, use Python 3.12 and install the recorded dependencies:

```bash
conda create -n tfmdehaze python=3.12 -y
conda activate tfmdehaze
python -m pip install -r requirements.txt
```

GPU execution requires a CUDA-capable PyTorch installation compatible with your system. CPU execution is also supported. The dependency file records the versions used locally; package availability and CUDA compatibility should be checked for a fresh installation.

## Run

From the repository root:

```bash
python dehazetab.py \
  --input-dir testdata/inputs \
  --output-dir testdata/outputs \
  --table-dir tables/table_method01_handcrafted
```

The default device is `auto`: CUDA when available, otherwise CPU. To select a device explicitly, append `--device cpu` or `--device cuda`.

Each result is saved as `<input_stem>_dehazed.png`, at the original input dimensions. Only final images are saved; intermediate transmission maps and query tables are not written. Supported input formats include PNG, JPEG, BMP, TIFF, and WebP. Input filenames must have unique stems.

### Main Options

| Argument | Default | Description |
|---|---|---|
| `--input-dir` | Required | Folder of hazy images |
| `--output-dir` | Required | Separate folder for dehazed PNGs |
| `--table-dir` | `tables/table_method01_handcrafted` | Reference CSVs and normalization metadata |
| `--index` | `tables/embeddings/resnet50.npz` | Matching retrieval index |
| `--device` | `auto` | `auto`, `cpu`, or `cuda` |
| `--num-references` | `5` | Number of retrieved context tables |
| `--patches-per-image` | Metadata, or `100` for legacy tables | Validates reference row count; does not subsample tables |
| `--batch-size` | `128` | Query rows per prediction call |
| `--threads` | `4` | CPU thread setting |

Run `python dehazetab.py --help` for all options, including guided-filter and cache settings. User-supplied relative paths are resolved from the working directory; default table, index, and cache paths are anchored to the script directory.

## Reference Data and Model Weights

The retrieval index, reference CSVs, and normalization metadata must correspond to the same reference collection. Original reference images are not needed at inference time; retrieved filenames identify the saved tables.

ResNet-50 and TabPFN v2 use pretrained weights. Local caches are stored under `.cache/torch/` and `.cache/tabpfn/`. If the weights are absent, first use may require network access and any authentication or access approval required by the model provider. Preserve the caches for offline use after setup.

When publishing, keep model caches, credentials, and generated outputs out of ordinary Git commits. Distribute large checkpoints separately only where their licenses permit, and ensure you have permission to redistribute the reference data and example images.

## Timing and Verification

`Proctime` is the full per-image processing time, including retrieval, feature extraction, TabPFN processing, refinement, and PNG saving. It is **not TabPFN-only inference time**; initial retrieval-encoder loading is outside the timer.

The standalone command was tested in `tfmdehaze` on CPU with eight example images. All eight completed successfully, with output dimensions matching their inputs.
