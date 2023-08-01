import pandas as pd
import os

def read_trace(path):
    tmp = pd.read_csv(path, header=None).values.tolist()
    return list(map(list, zip(*tmp)))

def find_first_peak(fp, n, m):
    fp_first_peak = []
    for i in range(n):
        dic = {}
        for key in fp[i]:
            dic[key] = dic.get(key, 0) + 1
        all_peak = []
        for i in sorted(dic):
            all_peak.append((i, dic[i]))
        
        cur_first_peak = []
        for i in range(1, len(all_peak)):
            if all_peak[i][1] >= all_peak[i - 1][1] and all_peak[i][1] >= all_peak[i + 1][1]:
                cur_first_peak = [all_peak[i - 1], all_peak[i], all_peak[i + 1]]
                break
        fp_first_peak.append(cur_first_peak[1][0])
    return fp_first_peak

def div_baseline(file_path: str, cur_time: str):
    # cpu
    for i, dir in enumerate(os.listdir(f'{file_path}/baseline')):
        if i == 0:
            no_load = pd.read_csv(f'{file_path}/baseline/{dir}/cpu.csv', header=None)
        else:
            no_load = pd.concat([no_load, pd.read_csv(f'{file_path}/baseline/{dir}/cpu.csv', header=None)], axis=0)

    no_load = no_load.values.tolist()
    no_load = list(map(list, zip(*no_load)))

    cpu_baseline = find_first_peak(no_load, len(no_load), len(no_load[0]))

    trace_cpu = read_trace(f'{file_path}/{cur_time}/cpu.csv')
    for i, b in zip(range(len(trace_cpu)), cpu_baseline):
        for j in range(len(trace_cpu[i])):
            trace_cpu[i][j] /= b
            trace_cpu[i][j] = float("{:.6f}".format(trace_cpu[i][j]))

    trace_cpu = list(zip(*trace_cpu))
    with open(f'{file_path}/{cur_time}/cpu_baseline.csv', 'w') as f:
        for l in trace_cpu:
            for d in l[:-1]:
                f.write("{:.06f},".format(d))
            f.write("{:.06f}\n".format(l[-1]))
    
    # gpu
    no_load = None
    for i, dir in enumerate(os.listdir(f'{file_path}/baseline')):
        if i == 0:
            no_load = pd.read_csv(f'{file_path}/baseline/{dir}/gpu.csv', header=None)
        else:
            no_load = pd.concat([no_load, pd.read_csv(f'{file_path}/baseline/{dir}/gpu.csv', header=None)], axis=0)

    no_load = no_load.values.tolist()
    no_load = list(map(list, zip(*no_load)))
    gpu_baseline = find_first_peak(no_load, len(no_load), len(no_load[0]))
    
    trace_gpu = read_trace(f'{file_path}/{cur_time}/gpu.csv')
    for i, b in zip(range(len(trace_gpu)), gpu_baseline):
        for j in range(len(trace_gpu[i])):
            trace_gpu[i][j] /= b
            trace_gpu[i][j] = float("{:.6f}".format(trace_gpu[i][j]))
    
    trace_gpu = list(zip(*trace_gpu))
    with open(f'{file_path}/{cur_time}/gpu_baseline.csv', 'w') as f:
        for l in trace_gpu:
            for d in l[:-1]:
                f.write("{:.06f},".format(d))
            f.write("{:.06f}\n".format(l[-1]))
    
