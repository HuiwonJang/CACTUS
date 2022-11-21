## Data setup
1. download `MiniImagenet` dataset from [here](https://lyy.mpi-inf.mpg.de/mtl/download/)
2. extract it like:
```shell
data/miniimagenet/
├── setup.py
├── test.tar
├── val.tar
└── train.tar
```
3. run setup.py

## Install

```bash
conda create -n unsup_meta python=3.9
conda activate unsup_meta
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
conda install ignite -c pytorch
conda install -c conda-forge faiss-cpu
pip install packaging tensorboard sklearn
```