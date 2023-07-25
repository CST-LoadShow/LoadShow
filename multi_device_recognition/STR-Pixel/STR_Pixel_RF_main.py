import gc
import logging
import os

from utils.get_feature_csv import getCsv, getCsvChoose
from utils.program_class import program_class, program_name
from random_forest import modelTrain
import warnings
warnings.filterwarnings("ignore")

if __name__ == "__main__":

    # get feature csv
    print("=========== get feature csv ===========")
    feature_list = ['mean', 'std', 'max', 'min', 'range', 'CV', 'RMS', 'MAD', 'skew', 'kurt',
                    'Q1', 'Median', 'Q3', 'IQR', 'SF', 'IF', 'CF']
    s = '../../dataset/div_firstpeak/Pixel'
    file_label = os.listdir(s + '/cpu')
    tmp = file_label[0].index("-")
    file_label = [s[tmp+1:] for s in file_label]

    file = s
    save_file = "STR_Pixel.csv"
    labels = getCsv(file_label, file, 64, save_file, 16, feature_list)
    print('labels =', labels)

    # random forest train
    print("=========== train random forest ===========")
    message = modelTrain(save_file)

