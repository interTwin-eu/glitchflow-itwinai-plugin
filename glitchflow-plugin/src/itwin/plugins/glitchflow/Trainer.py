from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
import zipfile
import math
from tqdm.notebook import tqdm
from sklearn.model_selection import train_test_split
from os import listdir
import h5py as h5
import os
import multiprocessing
from tqdm import tqdm
from PIL import Image

import gwpy
from gwpy.timeseries import TimeSeries


if torch.cuda.is_available():
    device = 'cuda'
    
else:
    device = 'cpu'
    
    
from typing import Any, Dict, Literal, Optional, Tuple
from itwinai.torch.trainer import RayTorchTrainer, TorchTrainer