import os

import numpy as np
import pandas as pd

from feature_api import getFeature


def getCsv(file_path_list, length, save_path, size, feature_list, fp_list=["cpu", "gpu"]):
    feature_len = len(feature_list)
    cpu_fp = os.path.join(file_path_list, 'cpu_baseline.csv')
    gpu_fp = os.path.join(file_path_list, 'gpu_baseline.csv')
    cur_data = pd.read_csv(cpu_fp, header=None)
    cur_data = np.array(cur_data)
    
    feature_cpu = np.empty(shape=(0, 16 * feature_len))
    for x in range(cur_data.shape[0] // length):
        feature = np.empty(shape=0)
        for j in range(cur_data.shape[1]):
            feature = np.concatenate((feature, getFeature(cur_data[x * length:(x + 1) * length, j])))

        feature = np.expand_dims(feature, 0)
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
    s = []
    for hd in fp_list:
        for i in range(size):
            for j in range(feature_len):
                s.append(hd + str(i) + feature_list[j])
    head = s
    df = pd.DataFrame(data=res_data)
    df.columns = head
    df.to_csv(save_path, index=False)
    return res_data
