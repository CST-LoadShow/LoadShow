import os
import time
from run_model import run_func
from data_process import div_baseline
from main_open_world import function

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
    # app_type, program_name, action_name = function(f'./data/{device_name}/{cur_time}')
    # print (program_name, action_name)
    # print (f"[+] Recognition Result\n    Time: {cur_time}\n    Type: {app_type}\n    Application: {program_name}\n    Action: {action_name}\n")
    print (f"[{cur_time}] Recognition Result")
    if app_type:
        print (f"    Type: {app_type}")
    print (f"    Application: {program_name}")
    if action_name:
        print (f"    Action: {action_name}")
    print ()


if __name__ == "__main__":
    # Rerun div_baseline for all data in `./data/{device_name}`
    device_name = 'Pixel'
    for test_time in os.listdir(f'./data/{device_name}'):
        if test_time == "baseline":
            continue
        div_baseline(f'./data/{device_name}', test_time)
        run_func(f'./data/{device_name}/{test_time}')
