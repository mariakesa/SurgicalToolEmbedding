import numpy as np
import torch
import os
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler, random_split
import glob
import cv2
from tqdm import tqdm

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
        img_tensor = torch.from_numpy(img)
        img_tensor = img_tensor.permute(2, 0, 1)
        class_id = torch.tensor([class_id])
        return img_tensor, class_id

def calc_mean_std(train_loader):
    # placeholders
    psum    = torch.tensor([0.0, 0.0, 0.0])
    psum_sq = torch.tensor([0.0, 0.0, 0.0])

    # loop through images
    for inputs in tqdm(train_loader):
        #print(inputs)
        psum    += inputs[0].sum(axis        = [0, 2, 3])
        psum_sq += (inputs[0] ** 2).sum(axis = [0, 2, 3])
    size = (100, 100)
    return psum, np.sqrt(psum_sq), size

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
    
    mean, std, size= calc_mean_std(train_loader)

    return {'train_loader':train_set,'validation_loader':valid_set,'mean':mean,'std':std,'size':size}