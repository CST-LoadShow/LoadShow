import os
from itertools import permutations

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from random_forest import modelTrain, modelTest
from utils.feature_api import getFeature
from utils.get_feature_csv import getCsv, getCsvNp,getCsvNpSoftware
import warnings
warnings.filterwarnings("ignore")


def getCsvTemp(file_path_list, length, save_path, size, feature_list, fp_list=["cpu", "gpu"], input_file_name=["cpu.csv", "gpu.csv"]):
    feature_len = len(feature_list)
    cpu_fp = os.path.join(file_path_list, input_file_name[0])
    gpu_fp = os.path.join(file_path_list, input_file_name[1])
    cur_data = pd.read_csv(cpu_fp, header=None)
    cur_data = np.array(cur_data)
    # print (cur_data)
    feature_cpu = np.empty(shape=(0, 16 * feature_len))
    for x in range(cur_data.shape[0] // length):
        feature = np.empty(shape=0)
        for j in range(cur_data.shape[1]):
            feature = np.concatenate((feature, getFeature(cur_data[x * length:(x + 1) * length, j])))

        feature = np.expand_dims(feature, 0)
        # print (feature, feature_cpu)
        feature_cpu = np.concatenate((feature_cpu, feature), axis=0)

    feature_gpu = np.empty(shape=(0, 16 * feature_len))
    cur_data = pd.read_csv(gpu_fp, header=None)
    cur_data = np.array(cur_data)
    for x in range(cur_data.shape[0] // length):
        feature = np.empty(shape=0)
        for j in range(cur_data.shape[1]):
            feature = np.concatenate((feature, getFeature(cur_data[x * length:(x + 1) * length, j])))

        feature = np.expand_dims(feature, 0)
        feature_gpu = np.concatenate((feature_gpu, feature), axis=0)

    count = min(feature_cpu.shape[0], feature_gpu.shape[0])
    feature_cpu_gpu = np.hstack((feature_cpu[:count, :], feature_gpu[:count, :]))
    res_data = feature_cpu_gpu
    return res_data


def getDate(file1, file2, software_name):
    feature_list = ['mean', 'std', 'max', 'min', 'range', 'CV', 'RMS', 'MAD', 'skew', 'kurt',
                    'Q1', 'Median', 'Q3', 'IQR', 'SF', 'IF', 'CF']

    size = 16
    s = []
    for i in range(size):
        for j in range(len(feature_list)):
            s.append(str(i) + feature_list[j])
    s = s + s
    head = ['label'] + s
    data = getCsvNpSoftware(file1, 64, size, feature_list, software_name[0], 0, size_max=200)
    for i in range(1, len(software_name)):
        data_np = getCsvNpSoftware(file1, 64, size, feature_list, software_name[i], i, size_max=200)
        data = np.vstack([data, data_np])

    df = pd.DataFrame(data=data)
    df.columns = head
    df.to_csv(file2, index=False)
    return


def getRealWorldData(file, software_name):
    feature_list = ['mean', 'std', 'max', 'min', 'range', 'CV', 'RMS', 'MAD', 'skew', 'kurt',
                    'Q1', 'Median', 'Q3', 'IQR', 'SF', 'IF', 'CF']
    file_list = os.listdir(file)
    label = []
    dataset = np.empty(shape=(0, 16 * len(feature_list) * 2))
    for f in file_list:
        f_name = f.split('-')[0]
        if f_name in software_name:
            path = os.path.join(file, f)
            path_list = os.listdir(path)
            count = 0
            for p in path_list:
                feature_file = os.path.join(path, p, 'feature.csv')
                if os.path.exists(feature_file):
                    feature = pd.read_csv(feature_file)
                else:
                    feature = getCsvTemp(os.path.join(path, p), 64, '', 16, feature_list,
                                         input_file_name=['cpu_baseline.csv', 'gpu_baseline.csv'])
                dataset = np.vstack([dataset, feature])
                count += feature.shape[0]
            label += count * [[software_name.index(f_name)]]
    label = np.array(label)
    # print(dataset.shape, label.shape)
    last_data = np.hstack((label, dataset))
    return last_data


if __name__ == "__main__":
    name = ['baseline', 'AliyunNetdisk', 'BaiduNetdisk', 'bilibili', 'iQiYi', 'pr2023', 'qq_music', 'spotify', 'tencent_meeting', 'wechat', 'zoom']
    save_file = "../../dataset/div_firstpeak/9600k-2060-behavior"
    input_file = "./file/data.csv"
    real_world_file = '../../dataset/div_firstpeak/real-world/windows'
    label_list = []
    getDate(save_file, input_file, name)
    data_real_world = getRealWorldData(real_world_file, name)
    df = pd.DataFrame(data=data_real_world)
    df.to_csv('file/real_world_data.csv', index=False)

    my_traces_timer = pd.read_csv(input_file)
    my_traces_timer = np.vstack([my_traces_timer, data_real_world])
    my_traces_timer = pd.DataFrame(my_traces_timer)

    y = my_traces_timer.iloc[:, 0]
    X = my_traces_timer.iloc[:, 1:]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = modelTrain(X_train, y_train, 'model/model.pth')
    message, probility, f1 = modelTest(X_test, y_test, model, 'pic/pic', name)
    print(message)


