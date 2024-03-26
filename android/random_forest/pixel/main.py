import gc
import logging
import os
import sys
sys.path.append('..')
from utils.program_class import program_class, program_name
from utils.get_feature_csv import getCsv, getCsvChoose2

from RF import modelTrain
import pandas as pd

if __name__ == "__main__":

    # get feature csv
    print("=========== get feature csv ===========")
    feature_list = ['mean', 'std', 'max', 'min', 'range', 'CV', 'RMS', 'MAD', 'skew', 'kurt',
                    'Q1', 'Median', 'Q3', 'IQR', 'SF', 'IF', 'CF']
    file_cpu = "../../dataset/pixel-500-ffff"
    file_gpu = "../../dataset/pixel-500-ffff"

    file_label = labels = ['baseline', 'bilibili', 'candy', 'cloudmusic', 'facebook', 'genshin',
                           'instagram', 'messenger', 'qq_music', 'royal', 'snapchat', 'soundcloud', 
                            'spotify', 'subway', 'tiktok', 'tmeeting', 'twitch', 'twitter', 
                            'weibo', 'youtube', 'zoom']


    save_file_cpu = "pixel_feature_cpu.csv"
    save_file_gpu = "pixel_feature_gpu.csv"

    print("cpu")
    getCsvChoose2(file_label, file_cpu, 64, save_file_cpu, 16, feature_list, choose_cpu_gpu = "cpu", size_max=32)
    print("gpu")
    getCsvChoose2(file_label, file_gpu, 64, save_file_gpu, 16, feature_list, choose_cpu_gpu = "gpu", size_max=32)
    

    save_file = "pixel_feature.csv"
    my_traces_timer_cpu = pd.read_csv(save_file_cpu)
    my_traces_timer_gpu = pd.read_csv(save_file_gpu)
    my_traces_timer_cpu = my_traces_timer_cpu.iloc[:, :]
    my_traces_timer_gpu = my_traces_timer_gpu.iloc[:, 1:]
    my_traces_timer = pd.concat([my_traces_timer_cpu, my_traces_timer_gpu], axis=1)
    my_traces_timer.to_csv(save_file, index=False)
   
    print("=========== train random forest ===========")
    model_save = "feature.pickle"
    pic_save = 'matrix.png'
    log_save = 'log.txt'
    logging.basicConfig(level=logging.DEBUG,
                        filename=log_save,
                        filemode='w')
    label_text = labels
    message = modelTrain(save_file_cpu, model_save, pic_save, label_text)
    message = 'class' + str(label_text) + '\n' + message
    print(message)
    logging.info(message)
    gc.collect()