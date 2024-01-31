from utils.get_feature_csv import getCsv_p
from random_forest import modelTrain

if __name__ == "__main__":
    # get feature csv
    print("=========== get feature csv ===========")
    feature_list = ['mean', 'std', 'max', 'min', 'range', 'CV', 'RMS', 'MAD', 'skew', 'kurt',
                    'Q1', 'Median', 'Q3', 'IQR', 'SF', 'IF', 'CF']
    file_list = ["../../dataset/div_firstpeak/9600k-2060-200-fff",
                 "../../dataset/div_firstpeak/9600k-2060-2000-ffff",
                 "../../dataset/div_firstpeak/9600k-2060-5000-3ffff",
                 "../../dataset/div_firstpeak/9600k-2060"]
    name_list = ["9600k-2060-200-fff", "9600k-2060-2000-ffff", "9600k-2060-5000-3ffff", "9600k-2060-20000-fffff"]

    file_label = ['7zip', 'formatfactory', 'matlab', 'mpcbe', 'obs', 'qq_music', 'tencent_meeting', 'utorrent']
    save_file_list = ["file/feature1.csv", "file/feature2.csv", "file/feature3.csv", "file/feature4.csv"]
    print(file_label)
    labels = file_label
    for i in range(len(file_list)):
        file = file_list[i]
        save_file = save_file_list[i]
        getCsv_p(file_label, file, 64, save_file, 16, feature_list, size_max=32)

    # random forest train
    print("=========== train random forest ===========")
    for i in range(len(file_list)):
        print(name_list[i])
        save_file = save_file_list[i]
        model_save = "9600_feature" + str(i) + ".pickle"
        pic_save = 'matrix' + str(i) + '.png'
        label_text = file_label
        message = modelTrain(save_file, model_save, pic_save, label_text)

