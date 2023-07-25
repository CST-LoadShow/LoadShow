import os

import numpy as np
import torch
from sklearn.model_selection import KFold
from torch import nn
from torch.utils.data import Subset
from torchvision.models import resnet18

from cnn_train import resnet18_train, picMatrix
from utils.dataset import CustomTensorDataset
from utils.get_tensor_dataset import get_tensor
from utils.program_class import program_class, program_name
import warnings
warnings.filterwarnings("ignore")

if __name__ == "__main__":
    # get tensor dataset
    print("============= get tensor ===============")
    file_label = os.listdir('../../dataset/div_firstpeak/9600k-2060/cpu')
    tmp = file_label[0].index("-")
    file_label = [s[tmp + 1:] for s in file_label]

    file = "../../dataset/div_firstpeak/9600k-2060/"
    save_dataset = "dataset_9600.pth"
    save_label = "label_9600.pth"

    labels = get_tensor(file_label, file, 16, save_dataset, save_label)
    print('labels =', labels)
    # labels = ['7zip', 'AliyunNetdisk', 'altium_designer', 'audition', 'BaiduNetdisk', 'bandicam', 'bandizip',
    #           'baseline', 'bilibili', 'cloudmusic', 'csgo', 'formatfactory', 'hogwarts', 'huorong', 'iQiYi',
    #           'iZotopeRX8', 'lol', 'matlab', 'mpcbe', 'obs', 'potplayer', 'pr2023', 'qq_music', 'spotify', 'sunlogin',
    #           'switch_audio_convert', 'tencent_meeting', 'tencent_video', 'unity', 'utorrent', 'vlc', 'wechat',
    #           'winrar', 'zoom']

    # sort labels by category
    color = ['b', 'c', 'g', 'k', 'r', 'y', 'b', 'c', 'g', 'k', 'r', 'y', 'b', 'c', 'g', 'k', 'r', 'w', 'y']
    label_text = []
    label_index = []
    color_list = []
    n = 0
    temp = "baseline"
    for key in program_class:
        if key in labels and key in file_label:
            label_text.append(key)
            label_index.append(file_label.index(key))
            if program_class[key] != temp:
                n += 1
                temp = program_class[key]
            color_list.append(color[n])
    for i in range(len(label_text)):
        label_text[i] = program_name[label_text[i]]

    train_data = torch.load(save_dataset)
    train_label = torch.load(save_label)

    dataset = CustomTensorDataset((train_data, train_label), 1, transform=False, train=True)

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    acc_list = []
    for i, (train_index, test_index) in enumerate(kf.split(dataset)):
        print(f"============= train {i} fold  ===============")
        dataset_train = Subset(dataset, train_index)
        dataset_test = Subset(dataset, test_index)
        resnet18_train(dataset_train, dataset_test, 40, len(labels), k=i)

        model = resnet18(num_classes=len(labels), pretrained=False)
        model.conv1 = nn.Conv2d(2, 64, kernel_size=7, stride=2, padding=3, bias=False)
        model.load_state_dict(torch.load(f'model/model_k_fold_{i}.pth'))
        model.to('cuda')
        # torch.save(model.state_dict(), f'model/model_k_fold_{i}.pth')
        print("============= pic matrix   ===============")
        save_pic = f'matrix_{i}.png'
        acc = picMatrix(model, dataset_test, label_index, label_text, color_list, save_pic)
        print("++++++++++++++ acc ++++++++++++++")
        print(acc)
        acc_list.append(acc)
    print("============= 5-k fold acc  ===============")
    print(acc_list)
    print(np.mean(acc_list))
