# cosmos

Bayesian analysis of the co-localization single-molecule microscopy image data.

----------------------------------------------------------------------------------------------------

## Installing

### Installing a stable cosmos release

**Install from source:**
```sh
git clone https://github.com/ordabayevy/cosmos.git
cd cosmos
pip install .
```

### Installing cosmos latest branch

**Install from source:**
```sh
conda create --name cosmos python=3.7
conda activate cosmos
pip install versioneer
conda install pytorch torchvision cudatoolkit=10.2 -c pytorch
git clone https://github.com/ordabayevy/cosmos.git
cd cosmos
git checkout latest
pip install .
jupyter nbextension     enable --py --sys-prefix appmode
jupyter serverextension enable --py --sys-prefix appmode
```

## Citation
If you use cosmos, please consider citing:
