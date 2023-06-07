import os
import torch
from single_application_recognition_CNN.cnn_train import resnet18_train, picMatrix
from utils.dataset import CustomTensorDataset
from utils.get_tensor_dataset import get_tensor
from utils.program_class import program_class, program_name

if __name__ == "__main__":
    # get tensor dataset
    print("============= get tensor ===============")
    file_label = os.listdir('../dataset/div_firstpeak/9600k-2060/cpu')
    tmp = file_label[0].index("-")
    file_label = [s[tmp + 1:] for s in file_label]

    file = "../dataset/div_firstpeak/9600k-2060"
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

    print("============= train   ===============")
    train_data = torch.load(save_dataset)
    train_label = torch.load(save_label)

    dataset_train = CustomTensorDataset((train_data, train_label), 0.8, transform=False, train=True)
    dataset_test = CustomTensorDataset((train_data, train_label), 0.8, transform=False, train=False)

    model = resnet18_train(dataset_train, dataset_test, 40, len(labels))

    # model = resnet18(num_classes=len(labels), pretrained=False)
    # model.conv1 = nn.Conv2d(2, 64, kernel_size=7, stride=2, padding=3, bias=False)
    # model.load_state_dict(torch.load('./model/model.pth'))
    # model.to('cuda')
    print("============= pic matrix   ===============")
    save_pic = 'matrix.png'
    picMatrix(model, dataset_test, label_index, label_text, color_list, save_pic)

