import os
import pickle

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

from utils.feature_api import getFeature


def modelTrain(save_model, label_text, label_list, random_state):
    X_train = pd.DataFrame()
    y_train = pd.DataFrame()
    X_test = pd.DataFrame()
    y_test = pd.DataFrame()
    for j in range(len(label_list)):
        s = label_list[j].split('-')[1][1]
        # print(j, s)
        if s == '0':
            if j + 1 < len(label_list) and label_list[j + 1].split('-')[1][1] == 'a':
                my_traces_timer1 = pd.read_csv(label_list[j])
                my_traces_timer2 = pd.read_csv(label_list[j + 1])
                my_traces_timer = pd.concat([my_traces_timer1, my_traces_timer2], axis=0)
                y = my_traces_timer.iloc[:, 0]
                X = my_traces_timer.iloc[:, 1:]
                X_train_temp, X_test_temp, y_train_temp, y_test_temp = train_test_split(X, y, test_size=0.5,
                                                                                        random_state=random_state)
                X_train = pd.concat([X_train, X_train_temp], axis=0)
                X_test = pd.concat([X_test, X_test_temp], axis=0)
                y_train = pd.concat([y_train, y_train_temp], axis=0)
                y_test = pd.concat([y_test, y_test_temp], axis=0)
            else:
                my_traces_timer = pd.read_csv(label_list[j])
                y = my_traces_timer.iloc[:, 0]
                X = my_traces_timer.iloc[:, 1:]
                X_train_temp, X_test_temp, y_train_temp, y_test_temp = train_test_split(X, y, test_size=0.2,
                                                                                        random_state=random_state)
                X_train = pd.concat([X_train, X_train_temp], axis=0)
                X_test = pd.concat([X_test, X_test_temp], axis=0)
                y_train = pd.concat([y_train, y_train_temp], axis=0)
                y_test = pd.concat([y_test, y_test_temp], axis=0)
    forest100 = RandomForestClassifier(n_estimators=100, random_state=0)
    forest100.fit(X_train, y_train)

    y_pred = forest100.predict(X_test)

    message = "n_estimators=100\n" + \
              "Accuracy on training set: {:.3f}\n".format(forest100.score(X_train, y_train)) + \
              "random rate = {:d}\n".format(random_state) + \
              "Accuracy on val set: {:.3f}\n".format(forest100.score(X_test, y_test)) + \
              classification_report(y_test, y_pred)
    dic = classification_report(y_test, y_pred, output_dict=True)
    # print(dic)
    recall = []
    for i in range(15):
        recall.append(dic[str(i) + '.0']['recall'])
    # print(recall)
    # print(np.mean(recall))
    f = open(save_model, 'wb')
    pickle.dump(forest100, f)
    f.close()

    return message, recall


def modelTest(dataset_path, save_model, label_text, split=False):
    my_traces_timer = pd.read_csv(dataset_path)
    y = my_traces_timer.iloc[:, 0]
    X = my_traces_timer.iloc[:, 1:]
    if split:
        X_train, X, y_train, y = train_test_split(X, y, test_size=0.2, random_state=42)
    f = open(save_model, 'rb')
    model = pickle.load(f)
    f.close()

    y_pred = model.predict(X)
    # print(y_pred)
    name = []
    for _ in y_pred:
        name.append(label_text[int(_)])
    # print(name)

    return model.score(X, y)


def getCsvTest(file_label, f, length, save_path, size, feature_list, size_max=128):
    feature_n = len(feature_list)
    f_cpu = os.path.join(f, 'cpu')
    f_gpu = os.path.join(f, 'gpu')
    cpu_list1 = os.listdir(f_cpu)
    test_file_list = []
    s = []
    for i in range(size):
        for j in range(feature_n):
            s.append(str(i) + feature_list[j])
    s = s + s
    head = ['label'] + s
    gpu_list1 = os.listdir(f_gpu)
    for i in range(len(cpu_list1)):
        if len(cpu_list1[i].split('-')) != 3:
            continue
        [device, app, case] = cpu_list1[i].split('-')
        # print(device, app, case)
        if app not in file_label:
            continue
        path = os.path.join(f_cpu, cpu_list1[i])
        fs = os.listdir(path)
        feature_cpu = np.empty(shape=(0, 16 * feature_n))
        for k in range(len(fs)):
            if k >= size_max:
                break
            csv_f = os.path.join(path,  fs[k])
            data = pd.read_csv(csv_f, header=None)
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

        for k in range(len(fs)):
            if k >= size_max:
                break
            csv_f = os.path.join(path, fs[k])
            data = pd.read_csv(csv_f, header=None)
            data = np.array(data)

            for x in range(data.shape[0] // length):
                feature = np.empty(shape=0)
                for j in range(data.shape[1]):
                    feature = np.concatenate((feature, getFeature(data[x * length:(x + 1) * length:, j])))
                feature = np.expand_dims(feature, 0)
                feature_gpu = np.concatenate((feature_gpu, feature), axis=0)

        count = min(feature_cpu.shape[0], feature_gpu.shape[0])
        feature_cpu_gpu = np.hstack((feature_cpu[:count, :], feature_gpu[:count, :]))
        dataset = feature_cpu_gpu
        label = count * [[file_label.index(app)]]
        label = np.array(label)
        last_data = np.hstack((label, dataset))
        save_file = os.path.join(save_path, f'{app}-{case}.csv')
        test_file_list.append(save_file)
        df = pd.DataFrame(data=last_data)
        df.columns = head
        df.to_csv(save_file, index=False)
    return test_file_list
