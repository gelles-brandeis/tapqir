import os
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
import configparser
import logging

from cosmos.utils.glimpse_reader import GlimpseDataset

def ReadAoi(dataset=None, control=None, device=None, path=None):
    data = GlimpseDataset(dataset, "data", device, path)
    if control is not None:
        control = GlimpseDataset(control, "control", device, data.path)

    return data, control
