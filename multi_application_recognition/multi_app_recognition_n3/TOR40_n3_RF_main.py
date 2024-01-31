from multi_application_recognition_RF_all.multi_app_recognition_n3.random_forest import modelTrain, modelTest
from utils.get_feature_csv import get_csv_multi, getCsv

if __name__ == "__main__":

    # get feature csv
    print("=========== get feature csv ===========")
    feature_list = ['mean', 'std', 'max', 'min', 'range', 'CV', 'RMS', 'MAD', 'skew', 'kurt',
                    'Q1', 'Median', 'Q3', 'IQR', 'SF', 'IF', 'CF']

    file_label = ['7zip', 'altium_designer', 'bilibili', 'matlab', 'pr2023', 'tencent_meeting', 'unity', 'vlc']
    file = "../../../dataset/multi_app_RF/9600k-2060-tri_label-train_v9"
    file_test = "../../../dataset/multi_app_RF/9600k-2060-multi4_label_v2"
    save_file_train = "feature_train.csv"
    save_file_test = "TOR40_feature_test.csv"

    labels = getCsv(file_label, file, 64, save_file_train, 16, feature_list)
    get_csv_multi(file_label, file_test, 64, save_file_test, 16, feature_list, size_max=32)
    label_text = file_label

    print("=========== train random forest ===========")
    model_save = "feature.pickle"
    message = modelTrain(save_file_train, model_save)
    message = 'class' + str(label_text) + '\n' + message
    print(message)

    # random forest test
    print("=========== test random forest ===========")
    message = modelTest(save_file_test, model_save, label_text, 'pic.png')
    print(message)
