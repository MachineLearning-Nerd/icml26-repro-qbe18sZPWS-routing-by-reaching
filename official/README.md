# Routing by Reaching: Composition of Pre-trained GFlowNets for Multi-Objective Generation

Code for [Routing by Reaching: Composition of Pre-trained GFlowNets for Multi-Objective Generation](https://arxiv.org/abs/2602.21565).

Two independent codebases:
- `mols/` — fragment/atom-based GFN (SEH, QM9), built on upstream `gflownet @ v0.2.0`
- `grid/` — synthetic HyperGrid, built on `torchgfn 2.4.1`

Neither installs as a package; run scripts from inside each folder so the local source is picked up via the current working directory.

## Setup

- mols

```bash
conda create -n mol python=3.10 -y && conda activate mol
pip install torch==2.1.2
PYG_LINKS=https://data.pyg.org/whl/torch-2.1.2+cu121.html

git clone https://github.com/recursionpharma/gflownet.git ~/.cache/gflownet_v0.2.0
cd ~/.cache/gflownet_v0.2.0 && git checkout v0.2.0
pip install -r requirements/main-3.10.txt --find-links $PYG_LINKS
pip install -e . --find-links $PYG_LINKS

pip install botorch rdkit
```

- grid

```bash
conda create -n grid_final python=3.10 -y && conda activate grid_final
pip install torch==2.1.2 torchgfn==2.4.1
```

## Usage

See `mols/scripts/` and `grid/scripts/` for the train/eval entry points.
