import os
from run_model import run_func
from data_process import div_baseline

def save_fingerprint(device_name, cpu_fingerprint, gpu_fingerprint, cur_time):
    if not os.path.exists(f'./data/{device_name}'):
        os.mkdir(f'./data/{device_name}')
    if not os.path.exists(f'./data/{device_name}/{cur_time}'):
        os.mkdir(f'./data/{device_name}/{cur_time}')
    with open(f'./data/{device_name}/{cur_time}/cpu.csv', 'a+') as f:
        f.write(cpu_fingerprint)
    with open(f'./data/{device_name}/{cur_time}/gpu.csv', 'a+') as f:
        f.write(gpu_fingerprint)
    div_baseline(f'./data/{device_name}', cur_time)
    app_type, program_name, action_name = run_func(f'./data/{device_name}/{cur_time}')
    print (f"[{cur_time}] Recognition Result\n    Type: {app_type}\n    Application: {program_name}\n    Action: {action_name}\n")
