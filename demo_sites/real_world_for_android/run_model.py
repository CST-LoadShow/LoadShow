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

    # app_type = ['Browser', 'Video Players', 'Music Players', 'Music Players', 'Meeting Applications']
    app_type = [None, "Video Applications", "Game Applications", "Music Applications", "Video Applications"]

    # model_name = ['baseline', 'bilibili', 'qq_music', 'spotify', 'tencent_meeting']
    model_name = ['none', 'bilibili', 'clash', 'spotify', 'tiktok']
    name = ['Home', 'Bilibili', 'Clash Royal', 'Spotify', 'TikTok']
    label_list = [
        ['Chrome'], 
        ['Browse video', 'Watch video'], 
        ['Browse music', 'Play music'], 
        ['Browse music', 'Play music'],
        ['Screen sharing', 'Video call', 'Voice call']
    ]
    data = getCsv(file_path, length, save_path, size, feature_list, fp_list=["cpu", "gpu"])
    # data = pd.read_csv(save_path)
    data = pd.DataFrame(data).values
    program_num = modelTest(data, model_path)
    # print(pred)
    program_name = name[int(program_num[0])]
    # print(program_name)
    action_name = None
    if int(program_num[0]) == 0:
        return app_type[int(program_num[0])], program_name, None
    else:
        # action_model = f'./model/model_{model_name[int(program_num[0])]}.pth'
        # action_num = modelTest(data, action_model)
        # action_name = label_list[int(program_num[0])][int(action_num[0])]
        # print(action_name)
        return app_type[int(program_num[0])], program_name, action_name


if __name__ == "__main__":
    # file = 'bilibili'
    # function(file)
    print (run_func('./data/9600k/2023-06-04 01:36:35'))
