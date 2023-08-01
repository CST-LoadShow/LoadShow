import gc
import logging
import numpy as np
from random_forest import modelTrain, modelTest, getCsvTest
import warnings


warnings.filterwarnings("ignore")


if __name__ == "__main__":
    # get feature csv
    feature_list = ['mean', 'std', 'max', 'min', 'range', 'CV', 'RMS', 'MAD', 'skew', 'kurt',
                    'Q1', 'Median', 'Q3', 'IQR', 'SF', 'IF', 'CF']

    file = "../../dataset/div_firstpeak/9600k-2060-multi_tab"
    save_file = "csv/case.csv"

    file_label = ['7zip', 'altium_designer', 'audition', 'BaiduNetdisk', 'bilibili', 'csgo',
                   'formatfactory', 'huorong', 'matlab', 'obs', 'pr2023', 'qq_music',
                   'tencent_meeting', 'unity', 'vlc']

    test_file = "../../dataset/div_firstpeak/9600k-2060-multi_tab"
    save_path = 'file/'
    test_file_list = getCsvTest(file_label, test_file, 64, save_path, 16, feature_list, size_max=128)
    # print(test_file_list)

    print("=========== train random forest ===========")
    model_save = "multi_tab.pickle"
    train_file_list = []
    for i in range(len(test_file_list)):
        case = test_file_list[i].split('-')[1][1]
        if case == '0' or case == 'a':
            train_file_list.append(test_file_list[i])
    # print(train_file_list)
    random_num = 1357
    _, case0_recall = modelTrain(model_save, file_label, train_file_list, random_num)

    message = ''
    c0 = []
    c1 = []
    c2 = []
    flag = [0]*len(file_label)
    print("=========== test random forest ===========")
    for f in test_file_list:
        app = f.split('-')[0].split('/')[1]
        s = f.split('-')[1][1]
        index = file_label.index(app)

        if s == '0' or s == '3' or s == 'a':
            continue
        if flag[index] == 0:
            flag[index] = 1
            message += f'file/{app}-c0.csv' + '  '
            message += "{:f}\n".format(case0_recall[index])

        acc = modelTest(f, model_save, file_label)
        message += f + '  '
        message += "{:.3f}\n".format(acc)

        if s == '1':
            c1.append(acc)
        elif s == '2':
            c2.append(acc)
    print(message)
    print('case0 acc mean:', np.mean(case0_recall))
    # print(c1)
    print('case1 acc mean:', np.mean(c1))
    # print(c2)
    print('case2 acc mean:', np.mean(c2))
