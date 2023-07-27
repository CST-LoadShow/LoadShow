from pathlib import Path
import pandas as pd
import torch.utils.data as Data
from PIL import Image
import numpy as np
import torch
import random

classes = list(range(25))


class MyDataset(Data.Dataset):
    @staticmethod
    def get_label(label_path):
        if label_path is not None:
            df = pd.read_csv(label_path)
            class_dict = {label: i for i, label in enumerate(classes)}
            df['label'] = df['label'].apply(lambda x: class_dict[x])
            return list(df['label'].values)
        else:
            return None

    def __init__(self, img_dir='dataset/trainImages/', train=True, img_label=None, transform=None):
        self.img_path = list(Path(img_dir).glob('*.png'))
        self.img_path.sort(key=lambda x: int(x.name.split('.')[0]))
        self.img_label = self.get_label(img_label)
        if img_label is not None:
            num_train = int(0.8 * len(self.img_path))
            index_list = list(range(len(self.img_path)))
            random.seed(42)
            indexes = random.sample(index_list, num_train)
            if not train:
                indexes = list(set(index_list) - set(indexes))

            self.img_path = [self.img_path[index] for index in indexes]
            self.img_label = [self.img_label[index] for index in indexes]
        self.transform = transform

    def __getitem__(self, index):
        if self.img_label is not None:
            img = Image.open(self.img_path[index]).convert('RGB')
            # img = Image.open(self.img_path[index]).convert('L')
            label = np.array(self.img_label[index], dtype=int)
            if self.transform is not None:
                img = self.transform(img)
            return img, torch.from_numpy(label)
        else:
            img = Image.open(self.img_path[index]).convert('RGB')
            if self.transform is not None:
                img = self.transform(img)
            return img, torch.from_numpy(np.array([]))

    def __len__(self):
        return len(self.img_path)


class CustomTensorDataset(Data.Dataset):
    """TensorDataset with support of transforms."""

    def __init__(self, tensors, l, transform=None,  train=True):
        assert all(tensors[0].size(0) == tensor.size(0) for tensor in tensors)

        num_train = int(l * tensors[0].size(0))
        index_list = list(range(tensors[0].size(0)))
        random.seed(42)
        indexes = random.sample(index_list, num_train)
        if not train:
            indexes = list(set(index_list) - set(indexes))
        tensors0 = torch.index_select(tensors[0], 0, torch.tensor(indexes))
        tensors1 = torch.index_select(tensors[1], 0, torch.tensor(indexes))
        self.tensors = (tensors0, tensors1)
        self.transform = transform

    def __getitem__(self, index):
        x = self.tensors[0][index]
        y = self.tensors[1][index]
        if self.transform:
            x = self.transform(x)
        return x, y

    def __len__(self):
        return self.tensors[0].size(0)
