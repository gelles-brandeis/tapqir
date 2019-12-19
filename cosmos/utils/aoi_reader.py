import os
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
import configparser
import logging

from cosmos.utils.glimpse_reader import GlimpseDataset

def ReadAoi(dataset, device):
    data = GlimpseDataset(dataset, device=device)

    return data
