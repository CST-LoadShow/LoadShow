import os.path
import pickle

import pandas as pd

from get_feature_csv import getCsv


def modelTest(X_test, save_model):
    f = open(save_model, 'rb')
    forest100 = pickle.load(f)
    f.close()
    y_pred = forest100.predict(X_test)

    return y_pred


def run_func(file_path: str):
    length = 64
    size = 16
    model_path = './model/model.pth'
    save_path = os.path.join(file_path, 'feature.csv')
    feature_list = ['mean', 'std', 'max', 'min', 'range', 'CV', 'RMS', 'MAD', 'skew', 'kurt',
                    'Q1', 'Median', 'Q3', 'IQR', 'SF', 'IF', 'CF']
    app_type = ['Browser', 'Download Tools', 'Download Tools', 'Video Players', 'Video Players', 'Video Processing Applications', 'Music Players', 'Music Players', 'Meeting Applications', 'Meeting Applications', 'Meeting Applications']
    model_name = ['chrome', 'AliyunNetdisk', 'BaiduNetdisk', 'bilibili', 'iQiYi', 'pr2023', 'qq_music', 'spotify', 'tencent_meeting', 'wechat', 'zoom']
    name = ['Chrome', 'Aliyun Netdisk', 'Baidu Netdisk', 'Bilibili', 'iQiYi', 'Premiere Pro', 'QQ music', 'Spotify', 'Tencent Meeting',
            'WeChat', 'Zoom']
    label_list = [['None'], ['Download', 'File preview', 'Upload', 'Watch video'], ['Download', 'File preview', 'Upload', 'Watch video'],
                  ['Browse video', 'Watch video'], ['Browse video', 'Watch video'], ['Video export', 'Video preview'], ['Browse music', 'Play music'], ['Browse music', 'Play music'],
                  ['Screen sharing', 'Video call', 'Voice call'], ['Browse moment', 'Screen sharing', 'Video call', 'Voice call'], ['Screen sharing', 'Video call', 'Voice call']]

    data = getCsv(file_path, length, save_path, size, feature_list, fp_list=["cpu", "gpu"])
    data = pd.DataFrame(data).values
    program_num = modelTest(data, model_path)
    program_name = name[int(program_num[0])]

    if int(program_num[0]) == 0:
        return app_type[int(program_num[0])], program_name, None
    else:
        action_model = f'./model/model_{model_name[int(program_num[0])]}.pth'
        action_num = modelTest(data, action_model)
        action_name = label_list[int(program_num[0])][int(action_num[0])]
        return app_type[int(program_num[0])], program_name, action_name

