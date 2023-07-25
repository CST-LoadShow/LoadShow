import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import classification_report
from torchvision.models import resnet18
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch.nn as nn
import torch
from utils.dataset import CustomTensorDataset


def multi_test(model, test_data, test_label, classes, flag=False, load_path='./model/model.pth'):
    batch_size = 64
    channels = 2
    n_classes = classes
    cuda = True if torch.cuda.is_available() else False

    dataset_test = CustomTensorDataset((test_data, test_label), 0, transform=False, train=False)

    test_dataloader = DataLoader(dataset_test, batch_size=batch_size, shuffle=False)
    if flag is True:
        model = resnet18(num_classes=n_classes, pretrained=False)
        model.conv1 = nn.Conv2d(channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        model = model.to('cuda')
        model.load_state_dict(torch.load(load_path))
    model.to('cuda')

    FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
    LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor

    # ----------
    #  Test
    # ----------
    model.eval()
    pred_label = []
    y_label_one = []
    y_label_multi = []

    with torch.no_grad():
        for data, label in test_dataloader:
            real_imgs = Variable(data.type(FloatTensor))
            labels = Variable(label.type(LongTensor))
            pred = model(real_imgs)
            p = pred.data.cpu().numpy()
            gt = labels.data.cpu().numpy()
            for i in range(gt.shape[0]):
                y = gt[i, :]
                y_label_one.append(y[1])
                y_label_multi.append(y[0])
            for i in range(len(p)):
                sorted_index_array = np.argsort(p[i])
                pred_label.append(sorted_index_array[-1])
        count_one = 0
        for i in range(len(pred_label)):
            if pred_label[i] == y_label_one[i]:
                count_one += 1
        dic = classification_report(y_label_one, pred_label, output_dict=True)

        print("test one main program acc {}".format(count_one / len(pred_label)))
        print("precision: {}".format(dic['macro avg']['precision']))
        print("recall: {}".format(dic['macro avg']['recall']))
    return model


def confusion_matrix_mine(preds, labels, conf_matrix):
    preds = torch.argmax(preds, 1)
    for p, t in zip(preds, labels):
        conf_matrix[p, t] += 1
    return conf_matrix


def getMatrixMultiLabel(model, n_classes, test_data, test_label, save_pic, label_text, label_multi_text):
    dataset_test = CustomTensorDataset((test_data, test_label), 0, transform=False, train=False)
    test_dataloader = DataLoader(dataset_test, batch_size=64, shuffle=False)
    plt.rcParams["font.family"] = "Times New Roman"

    cuda = True if torch.cuda.is_available() else False
    FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

    conf_matrix = torch.zeros(n_classes, len(label_multi_text))
    with torch.no_grad():
        for step, (imgs, targets) in enumerate(test_dataloader):
            real_imgs = Variable(imgs.type(FloatTensor))
            out = model(real_imgs)
            real_label = targets[:, 0].squeeze()
            conf_matrix = confusion_matrix_mine(out.cpu(), real_label, conf_matrix)
            conf_matrix = conf_matrix.cpu()
    conf_matrix = np.array(conf_matrix.cpu())
    conf_matrix = conf_matrix.transpose()
    fig, ax = plt.subplots(figsize=(7, 10))
    plt.tick_params(width=0.3, length=1.3)

    re_sort_col = [6, 3, 1, 4, 2, 7, 5, 0]
    re_sort_row = [9, 10, 5, 6, 2, 3, 7, 4, 11, 8, 0, 1]
    conf_matrix = conf_matrix[re_sort_row, :]
    conf_matrix = conf_matrix[:,re_sort_col ]
    numsize = 10
    tagsize =12
    labelsize = 14

    for x in range(n_classes):
        for y in range(len(label_multi_text)):
            info = int(conf_matrix[y, x])
            ax.text(x, y, info,
                    verticalalignment='center',
                    horizontalalignment='center',
                    color="w" if info >= 32 else "black", size=numsize)
    im = ax.imshow(conf_matrix, cmap='Blues')
    label_text = [label_text[i] for i in re_sort_col]
    label_multi_text = [label_multi_text[i] for i in re_sort_row]
    ax.set_xticks(range(len(label_text)), size=0.3)

    ax.set_xticklabels(label_text, size=tagsize, rotation=45, ha="right", family='Times New Roman')
    ax.set_yticks(range(len(label_multi_text)), size=0.3)
    ax.set_yticklabels(label_multi_text, size=tagsize, family='Times New Roman')

    ax.spines['bottom'].set_linewidth(0.3)
    ax.spines['left'].set_linewidth(0.3)
    ax.spines['right'].set_linewidth(0.3)
    ax.spines['top'].set_linewidth(0.3)
    ax.set_xlabel('Predicted Label', fontsize=labelsize)
    ax.set_ylabel('True Label', fontsize=labelsize)
    plt.savefig(save_pic, dpi=500, bbox_inches='tight')


