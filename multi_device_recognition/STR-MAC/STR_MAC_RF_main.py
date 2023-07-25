import gc
import logging
import os

from utils.get_feature_csv import getCsvChoose, mergeCSV, getCsv
from utils.program_class import program_class, program_name
from random_forest import modelTrain
if __name__ == "__main__":

    # get feature csv
    print("=========== get feature csv ===========")
    feature_list = ['mean', 'std', 'max', 'min', 'range', 'CV', 'RMS', 'MAD', 'skew', 'kurt',
                    'Q1', 'Median', 'Q3', 'IQR', 'SF', 'IF', 'CF']
    file_label = os.listdir('../../dataset/div_firstpeak/Mac1/cpu')
    tmp = file_label[0].index("-")
    file_label = [s[tmp+1:] for s in file_label]

    file1 = "../../dataset/div_firstpeak/Mac1"
    file2 = "../../dataset/div_firstpeak/Mac2"
    save_file1 = "STR_MAC1.csv"
    save_file2 = "STR_MAC2.csv"
    save_file3 = "STR_MAC.csv"

    labels = getCsv(file_label, file1, 64, save_file1, 16, feature_list)
    getCsv(file_label, file2, 64, save_file2, 16, feature_list)
    mergeCSV([save_file1, save_file2], save_file3)
    print('labels', labels)
    # sort labels by category
    color = ['b', 'c', 'g', 'k', 'r', 'y', 'b', 'c', 'g', 'k', 'r', 'y', 'b', 'c', 'g', 'k', 'r', 'w', 'y']
    label_text = []
    label_index = []
    color_list = []
    n = 0
    temp = "baseline"
    for key in program_class:
        if key in labels and key in file_label:
            label_text.append(key)
            label_index.append(file_label.index(key))
            if program_class[key] != temp:
                n += 1
                temp = program_class[key]
            color_list.append(color[n])
    for i in range(len(label_text)):
        label_text[i] = program_name[label_text[i]]
    # random forest train
    print("=========== STR_MAC1 ===========")
    modelTrain(save_file1, 5)
    print("=========== STR_MAC2 ===========")
    modelTrain(save_file2, 5)
    print("=========== STR_MAC 5k-fold ===========")
    modelTrain(save_file3, 5)
    print("=========== STR_MAC 10k-fold ===========")
    modelTrain(save_file3, 10)

