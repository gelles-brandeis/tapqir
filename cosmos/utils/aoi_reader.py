import os
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
import configparser
import logging

from cosmos.utils.glimpse_reader import load_data, read_glimpse

def ReadAoi(path=None, control=None, device=None):
    try:
        data = load_data(path, dtype="test", device=device)
        if control is not None:
            control = load_data(path, dtype="control", device=device)
    except:
        data = read_glimpse(path, D=14, dtype="test", device=device)
        data.save(path)
        if control is not None:
            control = read_glimpse(control, D=14, dtype="control", device=device)
            control.save(path)

    return data, control
