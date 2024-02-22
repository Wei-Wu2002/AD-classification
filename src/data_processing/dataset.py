import os

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


class CustomDataset(Dataset):
    def __init__(self, data_dir, label):
        self.data_dir = data_dir
        self.file_list = os.listdir(data_dir)
        self.label = label

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        file_path = self.data_dir + '/' + self.file_list[idx]
        # print(file_path)
        sample = torch.load(file_path)

        # minmax normalized
        min_val = sample.min()
        max_val = sample.max()
        sample = (sample - min_val) / (max_val - min_val)
        resized_sample = F.interpolate(sample.unsqueeze(0), size=(224, 224), mode='bilinear',
                                       align_corners=False).squeeze(0)
        return resized_sample, self.label

    def add_sample(self, other_dataset):
        self.file_list.extend(other_dataset.file_list)
