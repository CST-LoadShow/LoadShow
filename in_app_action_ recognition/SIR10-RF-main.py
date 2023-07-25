import pandas as pd
from sklearn.model_selection import train_test_split
from random_forest import modelTrain, modelTest
from utils.get_feature_csv import getCsvNpSoftware2
import warnings

warnings.filterwarnings("ignore")


def getDate(file1, file2, software_name):
    feature_list = ['mean', 'std', 'max', 'min', 'range', 'CV', 'RMS', 'MAD', 'skew', 'kurt',
                    'Q1', 'Median', 'Q3', 'IQR', 'SF', 'IF', 'CF']
    file_label = ['altium_designer', 'vlc', 'bilibili', 'tencent_video', 'iQiYi', 'cloudmusic',
                  'qq_music', 'wechat', 'zoom', 'tencent_meeting', 'AliyunNetdisk', 'BaiduNetdisk', 'bandizip',
                  'winrar', 'obs', 'bandicam', 'huorong']
    size = 16

    s = []
    for i in range(size):
        for j in range(len(feature_list)):
            s.append(str(i) + feature_list[j])
    s = s + s
    head = ['label'] + s
    data_np, labels = getCsvNpSoftware2(file_label, file1, 64, size, feature_list, software_name, size_max=200)
    df = pd.DataFrame(data=data_np)
    df.columns = head
    df.to_csv(file2, index=False)
    return labels


if __name__ == "__main__":
    name = ['AliyunNetdisk', 'BaiduNetdisk', 'bilibili', 'iQiYi', 'pr2023', 'qq_music', 'spotify', 'tencent_meeting',
            'wechat', 'zoom']

    save_file = "../dataset/div_firstpeak/9600k-2060-behavior"
    input_file = []
    label_list = []
    for i in range(len(name)):
        input_file.append("./file/%s.csv" % name[i])
        label = getDate(save_file, input_file[i], name[i])
        label_list.append(label)
    print('label_list =', label_list)
    acc = []
    f1_global = []
    for k in range(len(input_file)):
        my_traces_timer = pd.read_csv(input_file[k])
        y = my_traces_timer.iloc[:, 0]
        X = my_traces_timer.iloc[:, 1:]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.6, random_state=3770)

        model = modelTrain(X_train, y_train, 'model/model_%s.pth' % name[k])
        message, probility = modelTest(X_test, y_test, model, 'pic/pic_%s' % name[k], label_list[k])
        print("==================   %s   ==================" % name[k])
        print(message)
