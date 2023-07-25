import os
from itertools import permutations

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from cross_device_recognition.random_forest import modelTrain, modelTest, modelTestSimple
from utils.get_feature_csv import getCsvNp
from utils.program_class import program_class
from itertools import combinations
import warnings
warnings.filterwarnings("ignore")


def getDate():
    feature_list = ['mean', 'std', 'max', 'min', 'range', 'CV', 'RMS', 'MAD', 'skew', 'kurt',
                    'Q1', 'Median', 'Q3', 'IQR', 'SF', 'IF', 'CF']

    file1 = "../dataset/program-fp-data/div_firstpeak/410-A4"
    file2 = "../dataset/program-fp-data/div_firstpeak/410-A5"
    file3 = "../dataset/program-fp-data/div_firstpeak/410-A6"
    file4 = "../dataset/program-fp-data/div_firstpeak/410-A7"
    file5 = "../dataset/program-fp-data/div_firstpeak/410-A8"
    file6 = "../dataset/program-fp-data/div_firstpeak/410-B7"

    size = 16
    # n = 1
    list_1 = os.path.join(file1, 'cpu')
    list_1 = os.listdir(list_1)
    list_1 = [s[list_1[0][5:].index('-') + 6:] for s in list_1]

    list_2 = os.path.join(file2, 'cpu')
    list_2 = os.listdir(list_2)
    list_2 = [s[list_2[0][5:].index('-') + 6:] for s in list_2]

    list_3 = os.path.join(file3, 'cpu')
    list_3 = os.listdir(list_3)
    list_3 = [s[list_3[0][5:].index('-') + 6:] for s in list_3]

    list_4 = os.path.join(file4, 'cpu')
    list_4 = os.listdir(list_4)
    list_4 = [s[list_4[0][5:].index('-') + 6:] for s in list_4]

    list_5 = os.path.join(file5, 'cpu')
    list_5 = os.listdir(list_5)
    list_5 = [s[list_5[0][5:].index('-') + 6:] for s in list_5]

    list_6 = os.path.join(file6, 'cpu')
    list_6 = os.listdir(list_6)
    list_6 = [s[list_6[0][5:].index('-') + 6:] for s in list_6]

    file_label = [j for j in list_1 if j != 'baseline']
    data_np1 = getCsvNp(file_label, file1, 64, size, feature_list, size_max=200)
    data_np2 = getCsvNp(file_label, file2, 64, size, feature_list, size_max=200)
    data_np3 = getCsvNp(file_label, file3, 64, size, feature_list, size_max=200)
    data_np4 = getCsvNp(file_label, file4, 64, size, feature_list, size_max=200)
    data_np5 = getCsvNp(file_label, file5, 64, size, feature_list, size_max=200)
    data_np6 = getCsvNp(file_label, file6, 64, size, feature_list, size_max=200)

    data_np = np.vstack([data_np1, data_np2, data_np3, data_np4])

    s = []
    for i in range(size):
        for j in range(len(feature_list)):
            s.append(str(i) + feature_list[j])
    s = s + s
    head = ['label'] + s
    print(head)
    df = pd.DataFrame(data=data_np)
    df.columns = head
    save_file = "./file/multi.csv"
    df.to_csv(save_file, index=False)

    df = pd.DataFrame(data=data_np1)
    df.columns = head
    save_file = "./file/A4.csv"
    df.to_csv(save_file, index=False)

    df = pd.DataFrame(data=data_np2)
    df.columns = head
    save_file = "./file/A5.csv"
    df.to_csv(save_file, index=False)

    df = pd.DataFrame(data=data_np3)
    df.columns = head
    save_file = "./file/A6.csv"
    df.to_csv(save_file, index=False)

    df = pd.DataFrame(data=data_np4)
    df.columns = head
    save_file = "./file/A7.csv"
    df.to_csv(save_file, index=False)

    df = pd.DataFrame(data=data_np5)
    df.columns = head
    save_file = "./file/A8.csv"
    df.to_csv(save_file, index=False)

    df = pd.DataFrame(data=data_np6)
    df.columns = head
    save_file = "./file/B7.csv"
    df.to_csv(save_file, index=False)

    data_np_test = np.vstack([data_np5, data_np6])
    df = pd.DataFrame(data=data_np_test)
    df.columns = head
    save_file = "./file/test.csv"
    df.to_csv(save_file, index=False)

    color = ['b', 'c', 'g', 'k', 'r', 'y', 'b', 'c', 'g', 'k', 'r', 'y', 'b', 'c', 'g', 'k', 'r', 'w', 'y']
    label_class = []
    int_class = []
    color_list = []

    n = 0
    temp = "baseline"
    for key in program_class:
        if key in file_label:
            label_class.append(key)
            int_class.append(file_label.index(key))
            if program_class[key] != temp: 
                n += 1
                temp = program_class[key]
            color_list.append(color[n])
    print('label_text1 =', label_class)
    print('label_index1 =', int_class)
    print('color_list1 =', color_list)
    return label_class, int_class, color_list


if __name__ == "__main__":
    file = ["./file/A4.csv", "./file/A5.csv", "./file/A8.csv", "./file/A7.csv", "./file/B7.csv", "./file/A6.csv"]
    # label_text, label_index, color_list = getDate()
    label_text = ['altium_designer', 'formatfactory', 'vlc', 'bilibili', 'tencent_video', 'iQiYi', 'cloudmusic',
                   'qq_music', 'wechat', 'zoom', 'tencent_meeting', 'AliyunNetdisk', 'BaiduNetdisk', 'bandizip',
                   'winrar', 'obs', 'bandicam', 'huorong']
    label_index = [1, 7, 14, 5, 13, 9, 6, 11, 15, 17, 12, 0, 2, 4, 16, 10, 3, 8]
    color_list = ['c', 'g', 'k', 'k', 'k', 'k', 'r', 'r', 'y', 'y', 'y', 'b', 'b', 'c', 'c', 'g', 'g', 'g']
    # one sample
    global_acc = []
    X_train, X_test, y_train, y_test, X_test_all, y_test_all = [], [], [], [], [], []
    for k in range(len(file)):
        my_traces_timer = pd.read_csv(file[k])
        y = my_traces_timer.iloc[:, 0]
        X = my_traces_timer.iloc[:, 1:]
        X_test_all.append(X)
        y_test_all.append(y)
        X_train_temp, X_test_temp, y_train_temp, y_test_temp = train_test_split(X, y, test_size=0.3, random_state=42)
        X_train.append(X_train_temp)
        X_test.append(X_test_temp)
        y_train.append(y_train_temp)
        y_test.append(y_test_temp)
    X_train_temp = X_test_all[0]
    y_train_temp = y_test_all[0]
    # print(X_train_temp.shape)
    for i in range(len(file) - 1):
        if i != 0:
            X_train_temp = pd.concat([X_train_temp, X_test_all[i]])
            y_train_temp = pd.concat([y_train_temp, y_test_all[i]])
        print("=================  i = %d  ================ " % i)
        forest100 = modelTrain(X_train_temp, y_train_temp, 'model/model_%d' % i)
        temp_acc = []
        for j in range(i + 1, len(file), 1):
            print("=============  j = %d  ===============" % j)
            message, acc = modelTest(X_test_all[j], y_test_all[j], forest100, 'pic/pic_%d_%d' % (i, j), label_text,
                                     label_index, color_list)
            temp_acc.append(acc)
        global_acc.append(temp_acc)
    print("=======================  one case acc list  ======================")
    print(global_acc)

    # get average acc
    global_acc = []
    for i in range(len(file) - 1):
        pai_list = list(combinations(list(range(len(file))), i + 1))  # C(6,i)
        print(f"++++++++++++++++++++++ C(6,{i}) ++++++++++++++++++++++")
        # print(pai_list)
        temp_acc = []
        for C in pai_list:
            X_train_temp, y_train_temp = pd.DataFrame(), pd.DataFrame()
            for c in C:
                X_train_temp = pd.concat([X_train_temp, X_test_all[c]])
                y_train_temp = pd.concat([y_train_temp, y_test_all[c]])
            forest100 = modelTrain(X_train_temp, y_train_temp, 'model/model_%d_temp' % i)
            for j in range(len(file)):
                if j not in C:
                    acc = modelTestSimple(X_test_all[j], y_test_all[j], forest100)
                    temp_acc.append(acc)

        global_acc.append(temp_acc)
    print("=======================  average acc list  ======================")
    print(global_acc)
    for i in range(len(global_acc)):
        print(f"N_d = {i+1}")
        print('mean acc', np.mean(global_acc[i]))
        print('std acc', np.std(global_acc[i]))
