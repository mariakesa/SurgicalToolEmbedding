import numpy as np
import torch
import os
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler,ConcatDataset
import glob
import cv2
from tqdm import tqdm
import sys
sys.path.insert(0, "/home/maria/Desktop/t-simcne")
from tsimcne.tsimcne import TSimCNE
from tsimcne.imagedistortions import TransformedPairDataset, get_transforms
from loader import generate_loaders
from tqdm import tqdm

dct=generate_loaders()

train_loader=dct['train_loader']
validation_loader=dct['validation_loader']
mean=dct['mean']
std=dct['std']
size=dct['size']

dataset_full = torch.utils.data.ConcatDataset(
    [train_loader, validation_loader]
)

# data augmentations for contrastive training
transform = get_transforms(
    mean,
    std,
    size=size,
    setting="contrastive",
)
# transform_none just normalizes the sample
transform_none = get_transforms(
    mean,
    std,
    size=size,
    setting="test_linear_classifier",
)

# datasets that return two augmented views of a given datapoint (and label)
dataset_contrastive = TransformedPairDataset(train_loader, transform)
dataset_visualize = TransformedPairDataset(dataset_full, transform_none)

# wrap dataset into dataloader
train_dl = torch.utils.data.DataLoader(
    dataset_contrastive, batch_size=30, shuffle=True
)
orig_dl = torch.utils.data.DataLoader(
    dataset_visualize, batch_size=30, shuffle=False
)

import time

start=time.time()
# create the object
tsimcne = TSimCNE(total_epochs=[500, 50, 250])
# train on the augmented/contrastive dataloader (this takes the most time)
tsimcne.fit(train_dl)
# fit the original images
Y, labels = tsimcne.transform(orig_dl)
end=time.time()

print('Time to train: ', end-start)
import pickle
with open('filename.pickle', 'wb') as handle:
    pickle.dump({'Y':Y,'labels':labels}, handle, protocol=pickle.HIGHEST_PROTOCOL)

