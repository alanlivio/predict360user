# predict360user

predict360user is a library that aims to help researchers to reproduce and develop models to predict user behavior in 360 videos, namely trajectory (or traject for short).

## Getting Started

- [Tutorial](tutorial.md): Check out our interactive tutorial showing how to load datasets, train, and evaluate prediction models.
- [GitHub Repository](https://github.com/alanlivio/predict360user): Access the source code, contribute, or report issues.

## Requirements

The project requirements are in [requirements.txt](requirements.txt). See below how to create a conda env for that:

```bash
conda create -n p3u python==3.9 -y
conda activate p3u
pip install -r requirements.txt
conda install -c conda-forge cudatoolkit=11.2 cudnn=8.1.0
```

## Basic Usage

To train and evaluate the `pos_only` model on the `david` dataset:

```bash
python -m predict360user.start_run dataset=david model=pos_only
```
