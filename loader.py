import numpy as np
import torch
import os
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler, random_split
import glob
import cv2
from tqdm import tqdm
import sys
sys.path.insert(0, "/home/maria/Desktop/t-simcne")
from tsimcne.tsimcne import TSimCNE
from tsimcne.imagedistortions import TransformedPairDataset, get_transforms

class GeneralSurgeryToolsDataset(Dataset):
    def __init__(self):
        self.imgs_path = '/media/maria/DATA1/Hospitools_Dataset_DSLR/General_Surgery/General_Major/'
        tools=os.listdir(self.imgs_path)
        folder_list=[]
        for im_p in tools:
            folder_list.append(glob.glob(self.imgs_path + im_p)[0])
        self.data = []
        cn=[]
        for folder in folder_list:
            for file in os.scandir(folder):
                class_name = file.path.split("/")[-2]
                self.data.append([file.path, class_name])
                cn.append(class_name)
        self.class_map = {}
        i=0
        for c in set(cn):
            self.class_map[c]=i
            i+=1
        print(self.class_map)
        self.img_dim = (100, 100)
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        img_path, class_name = self.data[idx]
        img = cv2.imread(img_path)
        img = cv2.resize(img, self.img_dim)
        class_id = self.class_map[class_name]
        #img_tensor = torch.from_numpy(img)
        #img_tensor = img_tensor.permute(2, 0, 1).numpy()
        class_id = torch.tensor([class_id]).numpy()
        from PIL import Image
        #im=Image.fromarray(img_tensor)
        #print(img.shape)
        img=torch.Tensor(img.transpose(2,0,1))
        #img=Image.fromarray(img)
        return img, class_id

def calculate_mean_std(dataloader):
    mean = torch.zeros(3).to(torch.float32)
    variance = torch.zeros(3).to(torch.float32)
    num_samples = 0

    for data, _ in dataloader:
        # Get the data tensor from the batch
        data = data[0]

        # Sum up the values of each channel
        mean[0] += data[..., 0].mean().item()
        mean[1] += data[..., 1].mean().item()
        mean[2] += data[..., 2].mean().item()

        # Sum up the squared values of each channel
        variance[0] += data[..., 0].pow(2).mean().item()
        variance[1] += data[..., 1].pow(2).mean().item()
        variance[2] += data[..., 2].pow(2).mean().item()

        # Increment the number of samples
        num_samples += data.shape[0]

    # Divide the sum by the number of samples to get the mean
    mean /= num_samples

    # Divide the sum of squared values by the number of samples and subtract the square of the mean to get the variance
    variance /= num_samples
    variance -= mean.pow(2)

    # Take the square root of the variance to get the standard deviation
    std = torch.sqrt(variance)

    return mean, std


def generate_loaders():
    train_set = GeneralSurgeryToolsDataset()
    batch_size = 30
    validation_split = .5
    shuffle_dataset = True
    random_seed= 42

    train_set_size = int(len(train_set) * 0.5)
    valid_set_size = len(train_set) - train_set_size
    train_set, valid_set = random_split(train_set, [train_set_size, valid_set_size])

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size)
    validation_loader = torch.utils.data.DataLoader(valid_set, batch_size=batch_size)
    
    mean, std = calculate_mean_std(train_loader)

    return {'train_loader':train_set,'validation_loader':valid_set,'mean':mean,'std':std,'size':(100,100)}

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
tsimcne = TSimCNE(total_epochs=[10, 50, 250])
# train on the augmented/contrastive dataloader (this takes the most time)
tsimcne.fit(train_dl)
# fit the original images
Y, labels = tsimcne.transform(orig_dl)
end=time.time()

print('Time to train: ', end-start)
import pickle
with open('filename.pickle', 'wb') as handle:
    pickle.dump({'Y':Y,'labels':labels}, handle, protocol=pickle.HIGHEST_PROTOCOL)

