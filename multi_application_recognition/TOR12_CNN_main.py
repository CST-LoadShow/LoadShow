import os
import torch

from cnn_train import resnet18_train
from utils.dataset import CustomTensorDataset
from utils.get_tensor_dataset import get_tensor
from cnn_test_multi_label import multi_test, getMatrixMultiLabel
from utils.get_tensor_multi_label import get_feature_multi

if __name__ == "__main__":
    print("============= get tensor ===============")
    # get multi label dataset
    global_list = os.listdir('../dataset/div_firstpeak/9600k-2060-multi_label/cpu')
    file_label = []
    for i in range(len(global_list)):
        tmp1 = global_list[i].index("-")
        tmp2 = global_list[i].rindex("-")
        if tmp1 == tmp2:
            continue
        label1 = global_list[i][tmp1 + 1:tmp2]
        label2 = global_list[i][tmp2 + 1:]
        if label1 not in file_label:
            file_label.append(label1)

    label_index = range(len(file_label))

    file = "../dataset/div_firstpeak/9600k-2060-multi_label"
    save_dataset_multi = "dataset_multi.pth"
    save_label_multi = "label_multi.pth"
    get_feature_multi(file_label, file, 16, save_dataset_multi, save_label_multi)

    # get one label dataset
    file = "../dataset/div_firstpeak/9600k-2060/"
    save_dataset = "dataset_9600.pth"
    save_label = "label_9600.pth"
    get_tensor(file_label, file, 16, save_dataset, save_label)
    print("============= train ===============")
    # train dataset
    train_data = torch.load(save_dataset)
    train_label = torch.load(save_label)
    dataset_train = CustomTensorDataset((train_data, train_label), 0.8, transform=False, train=True)
    dataset_val = CustomTensorDataset((train_data, train_label), 0.8, transform=False, train=False)
    model = resnet18_train(dataset_train, dataset_val, 2, len(file_label))
    # test multi label dataset
    print("============= test ===============")
    test_data = torch.load("dataset_multi.pth")
    test_label = torch.load("label_multi.pth")
    # model = multi_test(model, test_data, test_label, len(file_label), flag=True, load_path='model/model.pth')
    multi_test(model, test_data, test_label, len(file_label), flag=False, load_path='./model/model.pth')

    # pic matrix
    print("============= pic ===============")
    label_text = ['7-Zip', 'Altium Designer', 'Bilibili', 'Matlab', 'Premiere Pro', 'Tencent Meeting', 'Unity',
                  'VLC Player']

    label_multi_text = ['7-Zip\nBaidu Netdisk', '7-Zip\nSpotify', 'Altium Designer\nBaidu Netdisk',
                        'Altium Designer\nSpotify',
                        'Bilibili\nBaidu Netdisk', 'Matlab\nBaidu Netdisk', 'Matlab\nSpotify',
                        'Premiere Pro\nBaidu Netdisk',
                        'Tencent Meeting\nBaidu Netdisk', 'Unity\nBaidu Netdisk', 'Unity\nSpotify',
                        'VLC Player\nBaidu Netdisk']
    save_pic = 'matrix.png'
    getMatrixMultiLabel(model, len(file_label), test_data, test_label, save_pic,  label_text, label_multi_text)