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
git clone https://github.com/ordabayevy/cosmos.git
cd cosmos
git checkout latest

conda create --name cosmos python=3.7
conda activate cosmos
pip install .
pip install . -f https://download.pytorch.org/whl/torch_stable.html
```

## Citation
If you use cosmos, please consider citing:
