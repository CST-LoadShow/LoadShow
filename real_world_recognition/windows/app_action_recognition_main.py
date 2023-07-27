import os
from itertools import permutations

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from random_forest import modelTrain, modelTest
from utils.feature_api import getFeature
from utils.get_feature_csv import getCsv, getCsvNp, getCsvNpSoftware, getCsvNpSoftware2
import warnings
warnings.filterwarnings("ignore")


def getCsvTemp(file_path_list, length, save_path, size, feature_list, fp_list=["cpu", "gpu"],
               input_file_name=["cpu.csv", "gpu.csv"]):
    feature_len = len(feature_list)
    # dataset = np.empty(shape=(0, 16 * feature_len * len(fp_list)))
    # label = []
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
    file_label = ['altium_designer', 'vlc', 'bilibili', 'tencent_video', 'iQiYi', 'cloudmusic',
                  'qq_music', 'wechat', 'zoom', 'tencent_meeting', 'AliyunNetdisk', 'BaiduNetdisk', 'bandizip',
                  'winrar', 'obs', 'bandicam', 'huorong']
    size = 16
    # n = 1

    s = []
    for i in range(size):
        for j in range(len(feature_list)):
            s.append(str(i) + feature_list[j])
    s = s + s
    head = ['label'] + s

    data_np, labels = getCsvNpSoftware2(file_label, file1, 64, size, feature_list, software_name, size_max=200)
    df = pd.DataFrame(data=data_np)
    df.columns = head
    df.to_csv(file2, index=False)
    return labels


def getRealWorldAction(file, software_name, action_label):
    feature_list = ['mean', 'std', 'max', 'min', 'range', 'CV', 'RMS', 'MAD', 'skew', 'kurt',
                    'Q1', 'Median', 'Q3', 'IQR', 'SF', 'IF', 'CF']
    file_list = os.listdir(file)
    label = []
    dataset = np.empty(shape=(0, 16 * len(feature_list) * 2))

    for f in file_list:
        # print(f.split('-'))
        f_name = f.split('-')[0]
        if f_name == software_name:
            action = f.split('-')[1]
            path = os.path.join(file, f)
            path_list = os.listdir(path)
            # count = len(path_list)
            count = 0
            for p in path_list:
                feature_file = os.path.join(path, p, 'feature.csv')
                if os.path.exists(feature_file):
                    feature = pd.read_csv(feature_file)
                else:
                    feature = getCsvTemp(os.path.join(path, p), 64, '', 16, feature_list,
                                         input_file_name=['cpu_baseline.csv', 'gpu_baseline.csv'])
                # feature = feature.iloc[:, 1:]
                # print(feature.shape)
                dataset = np.vstack([dataset, feature])
                count += feature.shape[0]
            label += count * [[action_label.index(action)]]
            print(f_name, action_label.index(action))
    if not label:
        return None
    else:
        label = np.array(label)
        last_data = np.hstack((label, dataset))
        return last_data


if __name__ == "__main__":

    name = ['AliyunNetdisk', 'BaiduNetdisk', 'bilibili', 'iQiYi', 'pr2023', 'qq_music', 'spotify', 'tencent_meeting',
            'wechat', 'zoom']
    real_world_file = '../../dataset/div_firstpeak/real-world/windows'
    save_file = "../../dataset/div_firstpeak/9600k-2060-behavior"
    input_file = []
    label_list = []
    for i in range(len(name)):
        input_file.append("./file/%s.csv" % name[i])
        label = getDate(save_file, input_file[i], name[i])
        label_list.append(label)

    print("label_list =", label_list)
    # label_list = [['download', 'upload', 'video'], ['download', 'upload', 'video'], ['browse', 'play'],
    #               ['browse', 'play'], ['export', 'preview'], ['browse', 'play'], ['browse', 'play'],
    #               ['screen', 'video', 'voice'], ['moment', 'screen', 'video', 'voice'], ['screen', 'video', 'voice']]

    acc = []
    f1_global = []
    for k in range(len(input_file)):

        print(name[k], label_list[k])
        data_real_world = getRealWorldAction(real_world_file, name[k], label_list[k])

        my_traces_timer = pd.read_csv(input_file[k])
        y_ = my_traces_timer.iloc[:, 0]
        X_ = my_traces_timer.iloc[:, 1:]
        y = pd.DataFrame(data_real_world).iloc[:, 0]
        X = pd.DataFrame(data_real_world).iloc[:, 1:]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        X_train = pd.concat([pd.DataFrame(X_train.values), pd.DataFrame(X_.values)], axis=0)
        y_train = pd.concat([pd.DataFrame(y_train.values), pd.DataFrame(y_.values)], axis=0)

        model = modelTrain(X_train, y_train, 'model/model_%s.pth' % name[k])
        message, probility, f1 = modelTest(X_test, y_test, model, 'pic/pic_%s.png' % name[k], label_list[k])
        acc.append(probility)
        f1_global.append(f1)
        print("==================   %s   ==================" % name[k])
        print(message)
    print(acc)
    print(f1_global)
    print(np.mean(f1_global))


