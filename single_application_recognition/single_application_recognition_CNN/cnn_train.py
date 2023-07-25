import numpy as np
from matplotlib import pyplot as plt
from mlxtend.plotting import plot_confusion_matrix
from sklearn.metrics import classification_report, confusion_matrix
from torchvision.models import resnet18
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch.nn as nn
import torch


def resnet18_train(dataset_train, dataset_test, train_epoth, classes, flag=False, load_path='./model/model.pth', k=0):
    n_epochs = train_epoth
    batch_size = 64
    lr = 0.0002
    b1 = 0.5
    b2 = 0.999
    n_classes = classes
    channels = 2

    cuda = True if torch.cuda.is_available() else False
    dataloader = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(dataset_test, batch_size=batch_size, shuffle=False)
    # Loss functions
    loss = torch.nn.CrossEntropyLoss()
    model = resnet18(num_classes=n_classes, pretrained=False)
    model.conv1 = nn.Conv2d(channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
    if flag:
        model.load_state_dict(torch.load(load_path))
    model.to('cuda')
    # Optimizers
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(b1, b2))
    FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
    LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor

    # ----------
    #  Training
    # ----------
    global_acc = 0
    acc_epoch5 = 0
    max_acc = 0
    for epoch in range(n_epochs):
        acc, _loss = 0.0, 0.0
        model.train()
        for i, (imgs, labels) in enumerate(dataloader):
            # Configure input
            real_imgs = Variable(imgs.type(FloatTensor))
            labels = Variable(labels.type(LongTensor))
            optimizer.zero_grad()
            pred = model(real_imgs)
            d_loss = loss(pred, labels)
            p = np.concatenate([pred.data.cpu().numpy()], axis=0)
            gt = np.concatenate([labels.data.cpu().numpy()], axis=0)
            d_acc = np.mean(np.argmax(p, axis=1) == gt)
            acc += d_acc
            _loss += d_loss.item()
            d_loss.backward()
            optimizer.step()
        print("================================================")
        print(
            "[Epoch %d/%d]  [D loss: %f, acc: %d%%]  !!!!!!!!!!!!!"
            % (
                epoch, n_epochs, _loss / len(dataloader), 100 * acc / len(dataloader))
        )
        if True:
            acc = .0
            model.eval()
            with torch.no_grad():
                for data, label in val_dataloader:
                    real_imgs = Variable(data.type(FloatTensor))
                    labels = Variable(label.type(LongTensor))
                    pred = model(real_imgs)
                    p = np.concatenate([pred.data.cpu().numpy()], axis=0)
                    gt = np.concatenate([labels.data.cpu().numpy()], axis=0)
                    d_acc = np.mean(np.argmax(p, axis=1) == gt)
                    acc += d_acc
                print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                print("val acc", acc * 100 / len(val_dataloader))
                if acc / len(val_dataloader) > max_acc:
                    max_acc = acc / len(val_dataloader)
                    torch.save(model.state_dict(), f'model/model_k_fold_{k}.pth')
                    print("=================== save model %d" % epoch)
                global_acc += acc / len(val_dataloader)
                acc_epoch5 += acc / len(val_dataloader)
                if epoch % 5 == 4:
                    print('each 5 epoch acc', acc_epoch5 / 5)
                    acc_epoch5 = 0
    # torch.save(model.state_dict(), 'model/model_last.pth')
    # print("=================== save final model")
    print('global acc ', global_acc / n_epochs)
    # model = model.load_state_dict(torch.load(f'model/model_k_fold_{k}.pth'))  #
    return


def picMatrix(model, dataset_test, label_index, label_text, color_list, save_pic):
    val_dataloader = DataLoader(dataset_test, batch_size=64, shuffle=False)
    cuda = True if torch.cuda.is_available() else False
    FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
    num = 0
    pred_label = np.empty(shape=0)
    real_label = np.empty(shape=0)
    model.eval()
    with torch.no_grad():
        for step, (imgs, targets) in enumerate(val_dataloader):
            num += targets.size(0)
            targets = targets.squeeze()  # [50,1] ----->  [50]
            real_label = np.concatenate((real_label, np.array(targets)))
            real_imgs = Variable(imgs.type(FloatTensor))
            out = model(real_imgs)
            preds = torch.argmax(out, 1)
            preds = preds.squeeze()
            pred_label = np.concatenate((pred_label, np.array(preds.cpu())))

    dic = classification_report(real_label, pred_label, output_dict=True)
    acc = dic['accuracy']
    cm = confusion_matrix(real_label, pred_label, labels=label_index)
    plt.rcParams["font.family"] = "Times New Roman"
    fig, ax = plot_confusion_matrix(conf_mat=cm,
                                    show_absolute=True,
                                    show_normed=False,
                                    colorbar=False,
                                    class_names=label_text,
                                    figsize=(15, 15),
                                    cmap='Blues',
                                    )

    for i in range(len(label_text)):
        ax.get_xticklabels()[i].set_color(color_list[i])
        ax.get_yticklabels()[i].set_color(color_list[i])

    plt.xlabel("Predicted Label", fontsize=14)
    plt.ylabel("True Label", fontsize=14)
    plt.savefig(save_pic, dpi=300, bbox_inches='tight')
    return acc
