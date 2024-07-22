# ProtoGate: Prototype-based Neural Networks with Global-to-local Feature Selection for Tabular Biomedical Data (ICML 2024)

[![Arxiv-Paper](https://img.shields.io/badge/Arxiv-Paper-yellow)](https://arxiv.org/abs/2306.12330)
[![Poster](https://img.shields.io/badge/-Poster-yellow)](https://icml.cc/media/PosterPDFs/ICML%202024/35215.png?t=1719613791.1975074)
[![Video presentation](https://img.shields.io/badge/Youtube-Video%20presentation-red)](https://www.youtube.com/watch?v=21BedzrsvSg) 
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/release/python-390/) 

Official PyTorch implementation of ["ProtoGate: Prototype-based Neural Networks with Global-to-local Feature Selection for Tabular Biomedical Data (ICML 2024)"](https://icml.cc/virtual/2024/poster/35215) 

by [Xiangjian Jiang](https://silencex12138.github.io/), [Andrei Margeloiu](https://www.cst.cam.ac.uk/people/am2770), [Nikola Simidjievski](https://simidjievskin.github.io/), [Mateja Jamnik](https://www.cl.cam.ac.uk/~mj201/).

## Installing

### Local Machine

The below settings have been tested to work on: CUDA Version: 11.2 (NVIDIA-SMI 460.32.03).

```
<!-- Install the codebase -->
cd REPOSITORY
conda create python=3.9.0 --name protogate
conda activate protogate
pip install -r requirements.txt

<!-- Optionally, install lightgbm -->
# pip install lightgbm --install-option=--gpu --install-option="--opencl-include-dir=/usr/local/cuda/include/" --install-option="--opencl-library=/usr/local/cuda/lib64/libOpenCL.so"
```

### Google Colab

If the environmental settings cannot work on the local machine, we also provide environmental dependecies for Google Colab. Please follow these steps:

* Upload the associate notebook `protogate_colab.ipynb` to [Google Colab](https://colab.research.google.com/)

* Upload the associate codebase to [Google Drive](https://drive.google.com/drive/my-drive)

* Set the `project_path` in notebook to the path of uploaded codebase in Google Drive

  ```python
  # Set up the path of codebase
  project_path = '/path/to/codebase/'
  ```

* Execute the notebook cells in `Step1: Set up environment`

## Running experiments

We provide scripts for ProtoGate and benchmark methods to work on the real-world (e.g., the “meta-pam” dataset) and synthetic datasets. Below are three examples:

* ProtoGate on `meta-pam` dataset

  ```bash
  bash scripts/PROTOGATE/run_exp_protogate_real.sh
  ```

* ProtoGate on `Syn1` dataset

  ```python
  bash scripts/PROTOGATE/run_exp_protogate_syn.sh
  ```

* The hyperparameters can be changed in the script by passing different values.

  ```bash
  python src/run_experiment.py \
  	--model 'protogate' \
  	--dataset 'metabric-pam50__200' \
  	--metric_model_selection total_loss \
  	--lr 0.1 \
  	--protogate_lam_global 0.0002 \
  	--protogate_lam_local 0.001 \
  	--pred_k 3 \
  	--max_steps 8000 \
  	--protogate_gating_hidden_layer_list 200 \
  	--tags 'real-world' \
  	--disable_wandb
  ```

## FAQ

* **Q: Where to find other real-world datasets?**

  A: The other HDLSS datasets can be downloaded from the [source website](https://jundongl.github.io/scikit-feature/datasets), [open-source project](https://github.com/andreimargeloiu/wpfs); and the non-HDLSS datasets can be downloaded from the [TabZilla benchmark](https://github.com/naszilla/tabzilla).

* **Q: How to get the full log of training and evaluation process?**

  A: Please install `wandb` library and cancel out `--disable_wandb` when running the experiments.

* **Q: How to set up the path of files in Google Drive?**

  A: The only special step is to mount the Google Drive to Google Colab, and the following steps are the same as the local machine.
  
* **Q: Which license does this codebase follow?**

  A: This codebase will follow the Apache-2.0 license when ProtoGate is publicly available for community.