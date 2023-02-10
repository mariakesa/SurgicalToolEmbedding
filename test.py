import numpy as np
import torch
import os
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
import glob
import cv2
from tqdm import tqdm
import sys
sys.path.insert(0, "/home/maria/Desktop/t-simcne")
from tsimcne.tsimcne import TSimCNE
from tsimcne.imagedistortions import TransformedPairDataset, get_transforms