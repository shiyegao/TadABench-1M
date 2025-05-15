# Introduction

This is the official implementation of the MLP model for the TadABench-1M dataset.

The dataset is available at [TadABench-1M](https://huggingface.co/datasets/JinGao/TadABench-1M).

# Installation

You should first install the [uv](https://docs.astral.sh/uv/) package manager.

Then, you can install the dependencies by running the following command:

```bash
uv sync
```

# Usage

## Run with uv

You can run the following commands to train the MLP model on the TadABench-1M dataset.

```bash
uv run scripts/run.py --cfg_path config/NB1M_ood_MLP_ESM2-35M.py
uv run scripts/run.py --cfg_path config/NB1M_ood_MLP_ESMC-300M.py
uv run scripts/run.py --cfg_path config/NB1M_ood_MLP_NT-50M.py
```

It will be faster if you extract the embeddings from the backbone model and save them to the disk.
Currently, the embeddings are not included in this repository.
We may release the embeddings in the future.


## Run with python

You can also use ``python`` by first init the venv and install the dependencies.

```bash
uv venv --python 3.11
source .venv/bin/activate
uv sync

python scripts/run.py --cfg_path config/NB1M_ood_MLP_ESM2-35M.py
python scripts/run.py --cfg_path config/NB1M_ood_MLP_ESMC-300M.py
python scripts/run.py --cfg_path config/NB1M_ood_MLP_NT-50M.py
```


## Wandb

If you want to use wandb, you can set the `use_wandb` to `True` in the config file and set the `wandb_proj_name` and `entity` to your own wandb project and entity.