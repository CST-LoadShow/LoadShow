import os
import pandas as pd
import numpy as np
import torch


def get_tensor(file_label, f, size, dataset_path, label_path):

    gpu_np = np.empty(shape=0)
    cpu_np = np.empty(shape=0)

    f_cpu = os.path.join(f, 'cpu')
    f_gpu = os.path.join(f, 'gpu')
    cpu_list1 = os.listdir(f_cpu)
    _ = cpu_list1[0].index('-')
    cpu_list2 = [s[_+1:] for s in cpu_list1]
    gpu_list1 = os.listdir(f_gpu)

    label = []
    dataset = np.empty(shape=(0, 2, size * 2, size * 2))
    for i in range(len(cpu_list1)):
        if cpu_list2[i] not in file_label:
            continue
        path = os.path.join(f_cpu, cpu_list1[i])
        fs = os.listdir(path)
        data_cpu = np.empty(shape=(0, size*2, size*2))
        for f in fs:
            csv_f = os.path.join(path, f)
            data = pd.read_csv(csv_f, header=None)

            data = np.array(data)
            data = np.log(data)

            cpu_np = np.concatenate([cpu_np, data.reshape(data.shape[0] * data.shape[1], )])
            data1 = data[range(0, size * 2), :].transpose()
            data2 = data[range(size * 2, size * 4), :].transpose()
            data = np.vstack((data1, data2))
            data = np.expand_dims(data, 0)
            data_cpu = np.vstack([data_cpu, data])
        data_gpu = np.empty(shape=(0, size*2, size*2))
        path = os.path.join(f_gpu, gpu_list1[i])
        fs = os.listdir(path)
        for f in fs:
            csv_f = os.path.join(path, f)
            data = pd.read_csv(csv_f, header=None)
            data = np.array(data)
            data = np.log(data)
            gpu_np = np.concatenate([gpu_np, data.reshape(data.shape[0] * data.shape[1], )])
            data1 = data[range(0, size * 2), :].transpose()
            data2 = data[range(size * 2, size * 4), :].transpose()
            data = np.vstack((data1, data2))
            data = np.expand_dims(data, 0)
            data_gpu = np.vstack([data_gpu, data])

        count = min(len(data_cpu), len(data_gpu))
        for j in range(count):
            d_cpu = np.expand_dims(data_cpu[j], 0)
            d_gpu = np.expand_dims(data_gpu[j], 0)
            data = np.concatenate((d_cpu, d_gpu), axis=0)
            data = np.expand_dims(data, 0)
            dataset = np.concatenate((dataset, data), axis=0)
        label += count * [file_label.index(cpu_list2[i])]
    label = np.array(label)
    label = torch.tensor(label).long()
    dataset = torch.tensor(dataset)
    torch.save(dataset, dataset_path)
    torch.save(label, label_path)

    return cpu_list2
