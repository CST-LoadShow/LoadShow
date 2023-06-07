import gc
import logging
import numpy as np
import pandas as pd
from utils.get_feature_csv import  getCsvNp
from utils.program_class import program_name
from random_forest import modelTrain
if __name__ == "__main__":

    # get feature csv
    print("=========== get feature multi ===========")
    feature_list = ['mean', 'std', 'max', 'min', 'range', 'CV', 'RMS', 'MAD', 'skew', 'kurt',
                    'Q1', 'Median', 'Q3', 'IQR', 'SF', 'IF', 'CF']

    file1 = "../dataset/div_firstpeak/9600k-2060"
    file2 = "../dataset/div_firstpeak/e5-k2000"
    file3 = "../dataset/div_firstpeak/10700-550X"
    save_file = "multi.csv"
    file_label = ['altium_designer', 'pr2023', 'vlc', 'bilibili', 'iQiYi', 'tencent_video', 'cloudmusic', 'qq_music',
                  'zoom', 'tencent_meeting', 'wechat', 'AliyunNetdisk', 'BaiduNetdisk', 'winrar',
                  'bandizip', 'bandicam', 'obs', 'huorong']

    data_np1 = getCsvNp(file_label, file1, 64, 16, feature_list, size_max=64)
    data_np2 = getCsvNp(file_label, file2, 64, 16, feature_list, size_max=32)
    data_np3 = getCsvNp(file_label, file3, 64, 16, feature_list, size_max=32)

    data_np = np.vstack([data_np1, data_np2, data_np3])
    s = []
    for i in range(16):
        for j in range(len(feature_list)):
            s.append(str(i) + feature_list[j])
    s = s + s
    head = ['label'] + s
    df = pd.DataFrame(data=data_np)
    df.columns = head
    df.to_csv(save_file, index=False)

    # random forest train
    for i in range(len(file_label)):
        file_label[i] = program_name[file_label[i]]

    print("=========== train random forest ===========")
    model_save = "multi_device.pickle"
    pic_save = 'matrix.png'
    log_save = 'log_multi_device.txt'
    logging.basicConfig(level=logging.DEBUG,
                        filename=log_save,
                        filemode='w')

    message = modelTrain(save_file, model_save, pic_save, file_label)
    message = 'class' + str(file_label) + '\n' + message
    print(message)
    logging.info(message)
    gc.collect()
