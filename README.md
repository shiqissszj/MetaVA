# MetaVA

A Lightweight Deep Neural Network for Personalized Detecting Ventricular Arrhythmias from a Single-Lead ECG Device

## Project Structure & Module Organization
- `pre_train.py`: Pre-training entry point and training loop configuration.
- `finetune.py`: Fine-tuning on downstream tasks using saved checkpoints.
- `MAML.py`: Meta-learning (MAML) training routine and evaluation utilities.
- `dataprocess.py`, `prepocessing.py`: Data cleaning, transforms, and dataset preparation.
- `tran_models.py`: Model architectures and wrappers used across scripts.
- `args.py`: Centralized CLI argument definitions and defaults.
- `util.py`: Shared helpers (logging, seeding, misc utilities).

## Getting Started

### Environment Set Up

- Create env: `conda create -n metava python=3.9`
- Install deps: `pip install -r requirements.txt` 

### Preprocess Data

`python preprocessing.py`

### Pre-training

Pre-train: `python pre_train.py`

### Fine-tuning

Fine-tune: `python finetune.py`

