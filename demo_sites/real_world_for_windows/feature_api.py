import os as _os

import numpy as _np
import pandas as _pd


# The average value
def mean_calculator(data: list):
    return sum(data) / len(data)


# Standard Deviation, Measures the variation
def std_calculator(data: list):
    avg = mean_calculator(data)
    res = 0
    for d in data:
        res += (d - avg) ** 2
    return (res / len(data)) ** 0.5


# The maximum value
def max_calculator(data: list):
    return max(data)


# The minimun value
def min_calculator(data: list):
    return min(data)


# The difference of maximum and minimum
def range_calculator(data: list):
    return max(data) - min(data)


# The average of absolute
def absMean_calculator(data: list):
    pass


# Coefficient of Variation, The ratio of Standard Deviation and Mean, CV = std / mean
def CV_calculator(data: list):
    # print(mean_calculator(data))
    if mean_calculator(data) == 0:
        return std_calculator(data) *100
    return std_calculator(data) / mean_calculator(data)


# Root Mean Square, Measures the effective energy
def RMS_calculator(data: list):
    res = 0
    for d in data:
        res += d ** 2
    return (res / len(data)) ** 0.5


# Mean Absolute Deviation, Measures the asymmetry and peakedness
def MAD_calculator(data: list):
    res = 0
    avg = mean_calculator(data)
    for d in data:
        res += abs(d - avg)
    return res / len(data)


# Skewness
def skew_calculator(data: list):
    # res1 = 0
    # res2 = 0
    # avg = mean_calculator(data)
    # for d in data:
    #     res1 += (d - avg) ** 3
    #     res2 += (d - avg) ** 2
    # n = len(data)
    # res1 /= n
    # res2 /= n
    # return ((n * (n - 1)) ** 0.5 / (n - 2)) * (res1 / (res2 ** 1.5))
    s = _pd.Series(data)
    return s.skew()


# Kurtosis
def kurt_calculator(data: list):
    # res = 0
    # avg = mean_calculator(data)
    # for d in data:
    #     res += (d - avg) ** 4
    # res /= 
    s = _pd.Series(data)
    return s.kurt()


# The first quantile
def Q1_calculator(data: list):
    s = _pd.Series(data)
    return s.quantile(0.25)


# Median
def Median_calculator(data: list):
    s = _pd.Series(data)
    return s.quantile(0.5)


# The thrid quantile
def Q3_calculator(data: list):
    s = _pd.Series(data)
    return s.quantile(0.75)


# The difference of Q3 and Q1
def IQR_calculator(data: list):
    s = _pd.Series(data)
    return s.quantile(0.75) - s.quantile(0.25)


# Mean Cross Rate
def MCR_calculator(data: list):
    pass


# Q1 Cross Rate
def Q1CR_calculator(data: list):
    pass


# Median Cross
def MedCR_calculator(data: list):
    pass


# Q3 Cross Rate
def Q3CR_calculator(data: list):
    pass


# Shape Factor
def SF_calculator(data: list):
    if mean_calculator(data) == 0:
        return RMS_calculator(data) * 100
    return RMS_calculator(data) / mean_calculator(data)


# Impulsive Factor
def IF_calculator(data: list):
    if mean_calculator(data) == 0:
        return max(data)*100
    return max(data) / mean_calculator(data)


# Crest Factor
def CF_calculator(data: list):
    if mean_calculator(data) == 0:
        return max(data)*100
    return max(data) / RMS_calculator(data)


# The thrid quantile
def Q95_calculator(data: list):
    s = _pd.Series(data)
    return s.quantile(0.99)


def Q01_calculator(data: list):
    s = _pd.Series(data)
    return s.quantile(0.01)


def getFeature(data):
    # print(data)
    # data = data.tolist()
    feature = []
    feature.append(mean_calculator(data))
    feature.append(std_calculator(data))
    feature.append(max_calculator(data))
    feature.append(min_calculator(data))
    feature.append(range_calculator(data))
    # feature.append(absMean_calculator(data))
    # feature.append(absMean_calculator(data))
    feature.append(CV_calculator(data))
    feature.append(RMS_calculator(data))

    feature.append(MAD_calculator(data))
    feature.append(skew_calculator(data))
    feature.append(kurt_calculator(data))

    feature.append(Q1_calculator(data))
    feature.append(Median_calculator(data))
    feature.append(Q3_calculator(data))
    feature.append(IQR_calculator(data))

    feature.append(SF_calculator(data))
    feature.append(IF_calculator(data))
    feature.append(CF_calculator(data))

    feature = _np.array(feature)
    # print(feature)
    return feature


def calculator(f, file_label):
    f_cpu = _os.path.join(f, 'cpu')
    f_gpu = _os.path.join(f, 'gpu')
    cpu_list1 = _os.listdir(f_cpu)
    _ = cpu_list1[0].index('-')
    print(_)
    cpu_list2 = [s[_ + 1:] for s in cpu_list1]
    gpu_list1 = _os.listdir(f_gpu)
    # gpu_list2 = [s[9:] for s in gpu_list1]
    print(cpu_list2)
    # print(gpu_list2)
    m_cpu, m_gpu = 0, 0
    n_cpu, n_gpu = 1000, 1000
    # cpu_95, gpu_95 = 0, 0
    dataset_cpu, dataset_gpu = _np.empty(shape=(64, 16)), _np.empty(shape=(64, 16))
    for i in range(len(cpu_list1)):
        if cpu_list2[i] not in file_label:
            continue
        path = _os.path.join(f_cpu, cpu_list1[i])
        fs = _os.listdir(path)
        for f in fs:
            csv_f = _os.path.join(path, f)
            data = _pd.read_csv(csv_f, header=None)
            # print(csv_f)
            # 获取所有列，并存入一个数组中
            data = _np.array(data)
            # print(data.shape)
            dataset_cpu = _np.concatenate((dataset_cpu, data), axis=0)
            maximum_temp = _np.max(data)
            minimum_temp = _np.min(data)
            if maximum_temp > m_cpu:
                m_cpu = maximum_temp
            if minimum_temp < n_cpu:
                n_cpu = minimum_temp

        # print(dataset_cpu.shape)
        path = _os.path.join(f_gpu, gpu_list1[i])

        # print(path)
        fs = _os.listdir(path)
        for f in fs:
            feature = _np.empty(shape=0)
            csv_f = _os.path.join(path, f)
            data = _pd.read_csv(csv_f, header=None)
            data = _np.array(data)
            dataset_gpu = _np.concatenate((dataset_gpu, data), axis=0)
            maximum_temp = _np.max(data)
            minimum_temp = _np.min(data)
            if maximum_temp > m_gpu:
                m_gpu = maximum_temp
            if minimum_temp < n_gpu:
                n_gpu = minimum_temp
    dataset_cpu = dataset_cpu.reshape(-1)
    dataset_gpu = dataset_gpu.reshape(-1)
    # print("dataset_cpu", dataset_cpu.shape)
    dataset_cpu = dataset_cpu.tolist()
    dataset_gpu = dataset_gpu.tolist()

    cpu_95 = Q95_calculator(dataset_cpu)
    gpu_95 = Q95_calculator(dataset_gpu)
    cpu_01 = Q01_calculator(dataset_cpu)
    gpu_01 = Q01_calculator(dataset_gpu)
    # print("cpu_95", cpu_95)
    return m_cpu, m_gpu, n_cpu, n_gpu, cpu_95, gpu_95, cpu_01, gpu_01

