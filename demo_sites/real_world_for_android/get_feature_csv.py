import os

import numpy as np
import pandas as pd

from feature_api import getFeature

# def getCsv(file_path_list, length, save_path, size, feature_list, fp_list=["cpu", "gpu"]):
#     feature_len = len(feature_list)
#     dataset = np.empty(shape=(0, 16 * feature_len * len(fp_list)))
#     # label = []
#     for file_path in file_path_list:
#         # print (file_path)
#         if "cpu" in fp_list:
#             file_cpu = os.path.join(file_path, 'cpu')
#             feature_cpu = np.empty(shape=(0, 16 * feature_len))
#             for cpu_fp in os.listdir(file_cpu):
#                 cur_data = pd.read_csv(os.path.join(file_cpu, cpu_fp), header=None)
#                 cur_data = np.array(cur_data)
#                 # print (cur_data)
#                 for x in range(cur_data.shape[0] // length):
#                     feature = np.empty(shape=0)
#                     for j in range(cur_data.shape[1]):
#                         feature = np.concatenate((feature, getFeature(cur_data[x * length:(x + 1) * length, j])))
#
#                     feature = np.expand_dims(feature, 0)
#                     # print (feature, feature_cpu)
#                     feature_cpu = np.concatenate((feature_cpu, feature), axis=0)
#         if "gpu" in fp_list:
#             file_gpu = os.path.join(file_path, 'gpu')
#             feature_gpu = np.empty(shape=(0, 16 * feature_len))
#             for gpu_fp in os.listdir(file_gpu):
#                 cur_data = pd.read_csv(os.path.join(file_path, 'gpu', gpu_fp), header=None)
#                 cur_data = np.array(cur_data)
#                 for x in range(cur_data.shape[0] // length):
#                     feature = np.empty(shape=0)
#                     for j in range(cur_data.shape[1]):
#                         feature = np.concatenate((feature, getFeature(cur_data[x * length:(x + 1) * length, j])))
#
#                     feature = np.expand_dims(feature, 0)
#                     feature_gpu = np.concatenate((feature_gpu, feature), axis=0)
#         if fp_list == ["cpu", "gpu"]:
#             count = min(feature_cpu.shape[0], feature_gpu.shape[0])
#             feature_cpu_gpu = np.hstack((feature_cpu[:count, :], feature_gpu[:count, :]))
#         elif fp_list == ["cpu"]:
#             count = feature_cpu.shape[0]
#             feature_cpu_gpu = feature_cpu
#         elif fp_list == ["gpu"]:
#             count = feature_gpu.shape[0]
#             feature_cpu_gpu = feature_gpu
#         dataset = np.vstack([dataset, feature_cpu_gpu])
#         # label += count * [[file_path[file_path.rindex('/') + 1:]]]
#     # label = np.array(label)
#     # print (len(label), len(dataset), dataset.shape)
#     # print(label, dataset)
#     # print (dataset[0])
#     # res_data = np.hstack((label, dataset))
#     res_data = dataset
#     s = []
#     for hd in fp_list:
#         for i in range(size):
#             for j in range(feature_len):
#                 s.append(hd + str(i) + feature_list[j])
#     head = s
#     df = pd.DataFrame(data=res_data)
#     df.columns = head
#     df.to_csv(save_path, index=False)
#     return res_data


def getCsv(file_path_list, length, save_path, size, feature_list, fp_list=["cpu", "gpu"]):
    feature_len = len(feature_list)
    # dataset = np.empty(shape=(0, 16 * feature_len * len(fp_list)))
    # label = []
    cpu_fp = os.path.join(file_path_list, 'cpu_baseline.csv')
    gpu_fp = os.path.join(file_path_list, 'gpu_baseline.csv')
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
