import gc
import pickle
import logging

import numpy as np
import pandas as pd
import graphviz
from keras.metrics import accuracy
from matplotlib import pyplot as plt
from sklearn import tree
from sklearn.metrics import classification_report, confusion_matrix
from mlxtend.plotting import plot_confusion_matrix
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.ensemble import RandomForestClassifier

from utils.program_class import program_name
from sklearn.preprocessing import LabelBinarizer


def modelTrain(X_train, y_train, save_model):

    forest100 = RandomForestClassifier(n_estimators=100, random_state=0)
    forest100.fit(X_train, y_train)
    f = open(save_model, 'wb')
    pickle.dump(forest100, f)
    f.close()
    return forest100


def modelTest(X_test, y_test, forest100, save_pic, label_text):
    y_pred = forest100.predict(X_test)

    probility = forest100.score(X_test, y_test)

    message = 'testing n_estimators=100\n' + \
              "Accuracy on test set: {:.3f}".format(forest100.score(X_test, y_test)) + '\n' + \
              classification_report(y_test, y_pred) + '\n'
    dic = classification_report(y_test, y_pred, output_dict=True)
    f1 = dic['macro avg']['f1-score']
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plot_confusion_matrix(conf_mat=cm,
                                    show_absolute=True,
                                    show_normed=False,
                                    colorbar=False,
                                    class_names=label_text,
                                    figsize=(15, 15),
                                    cmap='Blues',
                                    )

    plt.xlabel("Predicted Label", fontsize=14)
    plt.ylabel("True Label", fontsize=14)
    plt.savefig(save_pic, dpi=300, bbox_inches='tight')  # , bbox_inches='tight'
    return message, probility, f1
