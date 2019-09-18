BayesianImage
=============

### View jupyter notebooks
* [pyro-GMM.ipynb](https://github.com/gelles-brandeis/BayesianImage/blob/pyro-Yerdos/pyro-GMM.ipynb) - applying Bayesian Mixture Model to simulated data. Inferring the height of the spot, the background level, and the probabilities of the hidden states.
* [pyro-HMM.ipynb](https://github.com/gelles-brandeis/BayesianImage/blob/pyro-Yerdos/pyro-HMM.ipynb) - Hidden Markov Model with 2 states.

### Install python and [pyro](https://pyro.ai/)
* First [download](https://www.anaconda.com/distribution/#download-section) and install Anaconda
* Create a new environment and activate it (recommended)
> conda create --name cosmos

> conda activate cosmos

* Install numpy, matplotlib, jupyter, and pyro
> conda install numpy, matplotlib, jupyter

* Install pytorch using the command from this [link](https://pytorch.org/)
* Then install pyro

> pip3 install pyro-ppl

To work with notebooks open jupyter and navigate to the notebook
> jupyter notebook

or

> jupyter lab

To deactivate the environment
> conda deactivate cosmos

### Specific Aims
- [ ] Aim 1a. Test classification model on real and simulated data
- [ ] Aim 1b. Refine classification model
- [ ] Aim 1c. Multi-wavelength data
- [ ] Poisson, Normal, Gamma noise models evidence
- [ ] Noise vs mean
- [ ] Real images of no spot and spot
- [ ] 

- [ ] Aim 2a.

### Features
- [x] Read batches of data from glimpse files
- [ ] Bayesian image classification
- [ ] Calculate uncertainties using Bayesian approach
- [ ] Prior information about the surface locations of target molecules
- [ ] Spot amplitude, diamtere, and proximity
- [ ] Time-independent (TIBSD) and time-dependent (HMM) analysis
- [ ] Multi-wavelength
- [ ] Automatically determine number of classes

### Problems (to be addressed)
- [ ] Low signal-to-noise ratio (S/N) stemming from the high background noise and short acquisition times
- [ ] Photobleaching
- [ ] Anomaly detection

### To do list
- [x] GMM
- [x] scale parameter
- [ ] HMM
- [x] Batch fitting/GPU
- [ ] scale images
- [ ] MCC
