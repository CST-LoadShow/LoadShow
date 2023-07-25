import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from utils.feature_api import getFeature


def getCsv_matrix(f, size, n, dataset_list, _file, maximum=200):

    f_cpu = os.path.join(f, 'cpu')
    f_gpu = os.path.join(f, 'gpu')
    cpu_list1 = os.listdir(f_cpu)
    _ = cpu_list1[0].index('-')
    cpu_list2 = [s[_ + 1:] for s in cpu_list1]
    gpu_list1 = os.listdir(f_gpu)
    gpu_list2 = [s[8:] for s in gpu_list1]

    dataset = np.empty(shape=(0, size * 2 * n))
    label = []
    for i in range(len(cpu_list1)):
        if cpu_list2[i] not in file_label:
            continue
        path = os.path.join(f_cpu, cpu_list1[i])
        fs = os.listdir(path)
        # print(path)
        data_cpu = np.empty(shape=(0, size * n))
        for j in range(len(dataset_list)):
            if j >= maximum:
                break
            csv_f = os.path.join(path, str(dataset_list[j]) + '.csv')
            data = pd.read_csv(csv_f, header=None)
            data = np.array(data)
            data = data.reshape(-1, size * n)
            data_cpu = np.vstack([data_cpu, data])
        data_gpu = np.empty(shape=(0, size * n))
        path = os.path.join(f_gpu, gpu_list1[i])
        fs = os.listdir(path)
        for j in range(len(dataset_list)):
            if j >= maximum:
                break
            csv_f = os.path.join(path, str(dataset_list[j]) + '.csv')
            data = pd.read_csv(csv_f, header=None)
            data = np.array(data)
            data = data.reshape(-1, size * n)
            data_gpu = np.vstack([data_gpu, data])
        count = min(len(data_cpu), len(data_gpu))
        data_cpu_gpu = np.hstack((data_cpu[:count, :], data_gpu[:count, :]))
        dataset = np.vstack([dataset, data_cpu_gpu])
        label += count * [[file_label.index(cpu_list2[i])]]

    label = np.array(label)
    last_data = np.hstack((label, dataset))
    head = ['label'] + list(range(size * 2 * n))
    df = pd.DataFrame(data=last_data)
    df.columns = head
    df.to_csv(_file, index=False)
    return cpu_list2


def getCsvFeature(f, size, dataset_list, _file, f_list, maximum=200):
    feature_n = len(f_list)
    f_cpu = os.path.join(f, 'cpu')
    f_gpu = os.path.join(f, 'gpu')
    cpu_list1 = os.listdir(f_cpu)
    _ = cpu_list1[0].index('-')

    cpu_list2 = [s[_ + 1:] for s in cpu_list1]
    gpu_list1 = os.listdir(f_gpu)

    dataset = np.empty(shape=(0, 16 * feature_n * 2))
    label = []
    for i in range(len(cpu_list1)):
        if cpu_list2[i] not in file_label:
            continue
        path = os.path.join(f_cpu, cpu_list1[i])
        fs = os.listdir(path)
        feature_cpu = np.empty(shape=(0, 16 * feature_n))
        for f in dataset_list:
            csv_f = os.path.join(path, str(f) + '.csv')
            data = pd.read_csv(csv_f, header=None)
            data = np.array(data)

            for x in range(data.shape[0] // size):
                feature = np.empty(shape=0)
                for j in range(data.shape[1]):
                    feature = np.concatenate((feature, getFeature(data[x * size:(x + 1) * size, j])))

                feature = np.expand_dims(feature, 0)
                feature_cpu = np.concatenate((feature_cpu, feature), axis=0)

        feature_gpu = np.empty(shape=(0, 16 * feature_n))
        path = os.path.join(f_gpu, gpu_list1[i])

        for f in dataset_list:
            csv_f = os.path.join(path, str(f) + '.csv')
            data = pd.read_csv(csv_f, header=None)
            data = np.array(data)
            for x in range(data.shape[0] // size):
                feature = np.empty(shape=0)
                for j in range(data.shape[1]):
                    feature = np.concatenate((feature, getFeature(data[x * size:(x + 1) * size:, j])))
                feature = np.expand_dims(feature, 0)
                feature_gpu = np.concatenate((feature_gpu, feature), axis=0)

        count = min(feature_cpu.shape[0], feature_gpu.shape[0])
        feature_cpu_gpu = np.hstack((feature_cpu[:count, :], feature_gpu[:count, :]))
        dataset = np.vstack([dataset, feature_cpu_gpu])
        label += count * [[file_label.index(cpu_list2[i])]]
    label = np.array(label)
    last_data = np.hstack((label, dataset))
    s = []
    for i in range(16):
        for j in range(feature_n):
            s.append(str(i) + f_list[j])
    s = s + s
    head = ['label'] + s
    df = pd.DataFrame(data=last_data)
    df.columns = head
    df.to_csv(_file, index=False)
    return cpu_list2


def modelTrain(train_dataset_path, test_dataset_path):
    my_traces_timer = pd.read_csv(train_dataset_path)
    y_train = my_traces_timer.iloc[:, 0]
    X_train = my_traces_timer.iloc[:, 1:]

    my_traces_timer_test = pd.read_csv(test_dataset_path)
    y_test = my_traces_timer_test.iloc[:, 0]
    X_test = my_traces_timer_test.iloc[:, 1:]

    forest100 = RandomForestClassifier(n_estimators=100, random_state=0)
    forest100.fit(X_train, y_train)
    test_acc = forest100.score(X_test, y_test)

    return test_acc


if __name__ == "__main__":
    # random choose 20% of data as test data
    E = np.arange(1, 129)
    np.random.shuffle(E)
    # E = [113, 41, 107, 106, 97, 32, 43, 70, 3, 51, 124, 24, 45, 40, 65, 46, 127, 108, 77, 30, 66, 120, 13, 33, 84, 86,
    #      20, 61, 112, 37, 102, 12, 92, 6, 54, 100, 31, 29, 83, 110, 75, 126, 53, 125, 94, 49, 57, 2, 82, 52, 104, 79,
    #      27, 25, 117, 56, 95, 96, 17, 115, 87, 80, 74, 55, 121, 64, 48, 8, 28, 62, 58, 116, 14, 91, 35, 72, 68, 123, 60,
    #      69, 81, 93, 118, 73, 99, 7, 38, 98, 76, 103, 36, 11, 23, 63, 119, 122, 16, 18, 4, 39, 10, 71, 85, 44, 114, 90,
    #      26, 105, 89, 22, 101, 19, 88, 109, 50, 67, 59, 47, 15, 111, 34, 21, 5, 1, 9, 42, 78, 128]

    file = "../../dataset/div_firstpeak/9600k-2060/"
    global_list = os.listdir('../../dataset/div_firstpeak/9600k-2060/cpu')
    tmp = global_list[0].index("-")
    global_list = [s[tmp + 1:] for s in global_list if s[tmp + 1:] != "c_program"]
    file_label = global_list
    # print(file_label)
    global_acc_no_feature = []
    global_acc_feature = []
    test_size = 26  # int (128 * 0.2)

    train_size = [8, 16, 32, 64, 88, 102]
    split_list = [8, 16, 32, 64]

    test_list = E[:test_size]

    # no feature
    print("==============  no feature test ==============")
    for j in range(len(split_list)):
        print(f'---------- split {split_list[j]} ------------')
        save_file_test = "./file/9600_test%d.csv" % split_list[j]
        labels = getCsv_matrix(file, 16, split_list[j], test_list, save_file_test)
        same_split_acc = []
        for k in range(len(train_size)):
            print(f'---------- train size {train_size[k]} ------------')
            save_file = "./file/9600_train_%d_%d.csv" % (train_size[k], split_list[j])
            train_list = E[test_size: train_size[k] + test_size]
            labels = getCsv_matrix(file, 16, split_list[j], train_list, save_file)
            acc = modelTrain(save_file, save_file_test)
            same_split_acc.append(acc)
            print("acc", acc)
        global_acc_no_feature.append(same_split_acc)
    print("==============  no feature test acc ==============")
    print(global_acc_no_feature)

    # have feature
    print("==============  have feature test ==============")
    feature_list = ['mean', 'std', 'max', 'min', 'range', 'CV', 'RMS', 'MAD', 'skew', 'kurt',
                    'Q1', 'Median', 'Q3', 'IQR', 'SF', 'IF', 'CF']
    for j in range(len(split_list)):
        print(f'---------- split {split_list[j]} ------------')
        save_file_test = "./file/9600_test_feature_%d.csv" % split_list[j]
        labels = getCsvFeature(file, split_list[j], test_list, save_file_test, feature_list)
        same_split_acc = []
        for k in range(len(train_size)):
            print(f'---------- train size {train_size[k]} ------------')
            save_file = "./file/9600_train_feature_%d_%d.csv" % (train_size[k], split_list[j])
            train_list = E[test_size: train_size[k] + test_size]
            labels = getCsvFeature(file, split_list[j],  train_list, save_file, feature_list)
            acc = modelTrain(save_file, save_file_test)
            same_split_acc.append(acc)
            print("acc", acc)
        global_acc_feature.append(same_split_acc)
    print("==============  have feature test acc ==============")
    print(global_acc_feature)