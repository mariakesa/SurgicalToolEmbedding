import numpy as np
import torch
import os
from torch.utils.data import Dataset, DataLoader
import glob
import cv2

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
        for c in cn:
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