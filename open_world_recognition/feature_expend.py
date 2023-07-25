import os
import random
import pandas as pd
from utils.feature_api import *


def getCsvOpenWord(f, length, save_path, size, f_list):
    feature_n = len(f_list)
    # print(feature_n)
    # cpu 16 + gpu 16
    # f_2 = os.listdir(f)
    f_cpu = os.path.join(f, 'cpu')
    f_gpu = os.path.join(f, 'gpu')
    cpu_list1 = os.listdir(f_cpu)
    _ = cpu_list1[0].index('-')
    print(_)
    cpu_list2 = [s[_ + 1:] for s in cpu_list1]
    gpu_list1 = os.listdir(f_gpu)
    print(cpu_list2)

    dataset_0 = np.empty(shape=(0, 16 * feature_n * 2))  # 正样本
    dataset_1 = np.empty(shape=(0, 16 * feature_n * 2))  # 未知负样本
    dataset_test = np.empty(shape=(0, 16 * feature_n * 2))  # 测试样本
    label_0 = []
    label_1 = []
    label_test = []
    for i in range(len(cpu_list1)):
        if cpu_list2[i] not in file_label and cpu_list2[i] not in unknow_test_label:
            # print(cpu_list2[i])
            continue
        path = os.path.join(f_cpu, cpu_list1[i])
        fs = os.listdir(path)
        feature_cpu = np.empty(shape=(0, 16 * feature_n))
        for f in fs:
            #  一个csv 文件
            csv_f = os.path.join(path, f)
            data = pd.read_csv(csv_f, header=None)
            # 获取所有列，并存入一个数组中
            data = np.array(data)

            for x in range(data.shape[0] // length):
                feature = np.empty(shape=0)
                for j in range(data.shape[1]):
                    feature = np.concatenate((feature, getFeature(data[x * length:(x + 1) * length, j])))

                feature = np.expand_dims(feature, 0)
                feature_cpu = np.concatenate((feature_cpu, feature), axis=0)

        feature_gpu = np.empty(shape=(0, 16 * feature_n))
        path = os.path.join(f_gpu, gpu_list1[i])
        fs = os.listdir(path)

        for f in fs:
            csv_f = os.path.join(path, f)
            data = pd.read_csv(csv_f, header=None)
            data = np.array(data)

            for x in range(data.shape[0] // length):
                feature = np.empty(shape=0)
                for j in range(data.shape[1]):
                    feature = np.concatenate((feature, getFeature(data[x * length:(x + 1) * length:, j])))
                feature = np.expand_dims(feature, 0)
                feature_gpu = np.concatenate((feature_gpu, feature), axis=0)

        count = min(feature_cpu.shape[0], feature_gpu.shape[0])
        print(count)
        feature_cpu_gpu = np.hstack((feature_cpu[:count, :], feature_gpu[:count, :]))
        print(cpu_list2[i], '===========================')
        if cpu_list2[i] in file_label:
            if cpu_list2[i] in unknow_label:
                label_temp = len(know_label)
                dataset_1 = np.vstack([dataset_1, feature_cpu_gpu])
                label_1 += count * [[label_temp]]
                print("label", label_temp)
            else:
                label_temp = know_label.index(cpu_list2[i])
                dataset_0 = np.vstack([dataset_0, feature_cpu_gpu])
                label_0 += count * [[label_temp]]
                print("label", label_temp)

        else:
            dataset_test = np.vstack([dataset_test, feature_cpu_gpu])
            label_test += count * [[len(know_label)]]
            print("label", len(know_label))

    s = []
    for i in range(size):
        for j in range(feature_n):
            s.append(str(i) + f_list[j])
    s = s + s
    head = ['label'] + s
    print(head)

    label_1 = np.array(label_1)
    last_data_1 = np.hstack((label_1, dataset_1))
    df = pd.DataFrame(data=last_data_1)
    df.columns = head
    df.to_csv(save_path[1], index=False)

    label_0 = np.array(label_0)
    last_data_0 = np.hstack((label_0, dataset_0))
    df = pd.DataFrame(data=last_data_0)
    df.columns = head
    df.to_csv(save_path[0], index=False)

    label_test = np.array(label_test)
    last_data_test = np.hstack((label_test, dataset_test))
    df = pd.DataFrame(data=last_data_test)
    df.columns = head
    df.to_csv(save_path[2], index=False)
    return cpu_list2


if __name__ == "__main__":
    feature_list = ['mean', 'std', 'max', 'min', 'range', 'CV', 'RMS', 'MAD', 'skew', 'kurt',
                    'Q1', 'Median', 'Q3', 'IQR', 'SF', 'IF', 'CF']

    global_list = os.listdir('../dataset/div_firstpeak/9600k-2060/cpu')
    tmp = global_list[0].index("-")
    global_list = [s[tmp + 1:] for s in global_list if s[tmp + 1:] != "c_program" and s[tmp + 1:] != 'baseline']
    print('global list', global_list)

    train_size = [5, 10, 15]

    rot8_label = ['unity', 'audition', 'potplayer', 'cloudmusic', 'zoom', 'AliyunNetdisk', 'winrar', 'csgo', 'obs']
    for u in range(len(rot8_label)):
        rest_list = [x for x in global_list if x != rot8_label[u]]
        test_list = random.sample(rest_list, 15)
        for v in range(len(train_size)):
            train_list = random.sample([x for x in rest_list if x not in test_list], train_size[v])
            know_label = [rot8_label[u]]
            file_label = [rot8_label[u]] + train_list
            unknow_label = train_list
            unknow_test_label = test_list
            real_label = [rot8_label[u], 'unknow']
            print('know_label', know_label)
            print('unknow_label', unknow_label)
            print('file_label', file_label)
            print('unknow_test_label', unknow_test_label)
            file = "../dataset/div_firstpeak/9600k-2060"
            save_file = ["./file/feature_%s_%d_0.csv" % (rot8_label[u], train_size[v]),
                         "./file/feature_%s_%d_1.csv" % (rot8_label[u], train_size[v]), "./file/test_%s.csv" % rot8_label[u]]
            labels = getCsvOpenWord(file, 64, save_file, 16, feature_list)

