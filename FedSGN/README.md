# FedSGN

FedSGN is a federated subspace learning and anomaly detection codebase. This repository contains the original framework, including the FedSGN server and client implementations.

## Repository structure

- `main.py` — command-line entry point.
- `FLAlgorithms/` — server, client, optimizer, and model implementations.
- `utils/` — data loading, optimization, evaluation, and plotting helpers.

## Requirements

Use Python 3 and install the packages listed in `requirements.txt`:

```bash
pip install -r requirements.txt
```

For GPU use, install a PyTorch and torchvision build compatible with your CUDA version.

## Data

Datasets are not included. The code expects CSV files under `abnormal_detection_data/train/` and `abnormal_detection_data/test/`. Dataset filenames are referenced directly by the data-loading and evaluation code, so place the required files there before running the program.

Supported dataset names in the command-line parser are `Unsw`, `Iot23`, `Ton`, `KDD`, and `syn`.

## Run

From the repository root:

```bash
python main.py --algorithm FedSGN --dataset Unsw
```

Use `python main.py --help` to see the available command-line options. The selected dataset files must be present in the expected folders.
