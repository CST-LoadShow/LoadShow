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


def modelTestExpend(X_test, save_model):
    f = open(save_model, 'rb')
    forest100 = pickle.load(f)
    f.close()
    y_pred = list(forest100.predict_proba(X_test))
    # y = forest100.predict(X_test)
    # print(y_pred.shape)
    # print(y_pred)
    return y_pred, 0 if y_pred[0][0] > 0.4 else 1


def function(file_path: str):
    length = 64
    size = 16
    # model_path = './model'
    save_path = os.path.join(file_path, 'feature.csv')
    feature_list = ['mean', 'std', 'max', 'min', 'range', 'CV', 'RMS', 'MAD', 'skew', 'kurt',
                    'Q1', 'Median', 'Q3', 'IQR', 'SF', 'IF', 'CF']
    
    app_type = ['Browser', 'Video Players', 'Music Players', 'Music Players', 'Meeting Applications']

    # name_file = ['baseline', 'bilibili', 'qq_music', 'spotify', 'tencent_meeting']
    name_file = ['bilibili', 'clash', 'spotify', 'tiktok']
    # model_name = ['chrome', 'AliyunNetdisk', 'BaiduNetdisk', 'bilibili', 'iQiYi', 'pr2023', 'qq_music', 'spotify', 'tencent_meeting', 'wechat', 'zoom']
    # name = ['Chrome', 'Bilibili', 'QQ music', 'Spotify', 'Tencent Meeting']
    name = ['Bilibili', 'Clash Royal', 'Spotify', 'TikTok']
    label_list = [
        ['Chrome'], 
        ['Browse video', 'Watch video'], 
        ['Browse music', 'Play music'], 
        ['Browse music', 'Play music'],
        ['Screen sharing', 'Video call', 'Voice call']
    ]

    data = getCsv(file_path, length, save_path, size, feature_list, fp_list=["cpu", "gpu"])
    data = pd.read_csv(save_path)
    # data = pd.read_csv(save_path)
    data = pd.DataFrame(data).values

    probably_list = []
    classes_list = []
    for u in range(len(name)):
        model_path = f'./model_open_world/model_{name_file[u]}.pth'
        pro, classes = modelTestExpend(data, model_path)  # 二分类每个分类的概率
        probably_list.append(pro)
        classes_list.append(classes)
    print(probably_list)
    print(classes_list)
    if classes_list.count(0) == 1:
        program_num = classes_list.index(0)
    elif classes_list.count(0) > 1:
        # 识别了多个软件
        probably_max = [list(probably_list[0][0])[0], 0]
        for v in range(len(classes_list)):
            if classes_list[v] == 0:
                # print(list(probably_list[v][0]))
                pro = list(probably_list[v][0])
                # print(pro[0])
                # print(probably_max[0])
                if pro[0] > probably_max[0]:
                    probably_max[0] = pro[0]
                    probably_max[1] = v

        program_num = probably_max[1]
    else:
        # 未识别出软件
        program_num = -1

    # print(pred)
    # print(program_num)
    # print(program_name)

    if program_num == -1:
        # print('Other app')
        return 'Unknown', 'Other App', None
    elif program_num == 0:
        # print('chrome')
        return 'Chrome', 'Chrome', None
    else:
        type_name = app_type[program_num]
        program_name = name[program_num]
        action_model = f'./model/model_{name_file[program_num]}.pth'
        action_num = modelTest(data, action_model)
        action_name = label_list[program_num][int(action_num[0])]
        # print(program_name)
        # print(action_name)
        return type_name, program_name, action_name


if __name__ == "__main__":
    file = 'bilibili'
    function(file)
