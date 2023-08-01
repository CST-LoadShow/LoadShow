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

    app_type = [None, "Video Applications", "Game Applications", "Music Applications", "Video Applications"]

    name = ['Home', 'Bilibili', 'Clash Royal', 'Spotify', 'TikTok']
    data = getCsv(file_path, length, save_path, size, feature_list, fp_list=["cpu", "gpu"])
    data = pd.DataFrame(data).values
    program_num = modelTest(data, model_path)
    program_name = name[int(program_num[0])]
    action_name = None
    if int(program_num[0]) == 0:
        return app_type[int(program_num[0])], program_name, None
    else:
        return app_type[int(program_num[0])], program_name, action_name

