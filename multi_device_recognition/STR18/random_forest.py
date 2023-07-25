import pickle

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.constants import lb
from sklearn.metrics import confusion_matrix
from mlxtend.plotting import plot_confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import LabelBinarizer


def modelTrain(dataset_path, save_model, save_pic, label_text):
    my_traces_timer = pd.read_csv(dataset_path)
    y = my_traces_timer.iloc[:, 0]
    X = my_traces_timer.iloc[:, 1:]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=300)

    forest100 = RandomForestClassifier(n_estimators=100, random_state=0)
    forest100.fit(X_train, y_train)
    y_pred = forest100.predict(X_test)

    message = "n_estimators=100\n" + \
              "Accuracy on training set: {:.3f}\n".format(forest100.score(X_train, y_train)) + \
              "Accuracy on val set: {:.3f}\n".format(forest100.score(X_test, y_test)) + \
              classification_report(y_test, y_pred)

    cm = confusion_matrix(y_test, y_pred)
    plt.rcParams["font.family"] = "Times New Roman"
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
    plt.savefig(save_pic, dpi=300, bbox_inches='tight')

    f = open(save_model, 'wb')
    pickle.dump(forest100, f)
    f.close()

    return message


def modelTrain5K(dataset_path):
    my_traces_timer = pd.read_csv(dataset_path)
    y = my_traces_timer.iloc[:, 0]
    X = my_traces_timer.iloc[:, 1:]
    # y = pd.DataFrame(y)
    lb = LabelBinarizer()
    y_ = np.array([number[0] for number in lb.fit_transform(y)])

    forest_5k = RandomForestClassifier(n_estimators=100, random_state=0)
    kfold = KFold(n_splits=5, shuffle=True, random_state=9)

    acc_scores = cross_val_score(forest_5k, X, y, cv=kfold, scoring="accuracy")
    pre_scores = cross_val_score(forest_5k, X, y_, cv=kfold, scoring="precision_macro")
    recall_scores = cross_val_score(forest_5k, X, y_, cv=kfold, scoring="recall_macro")

    print('acc_scores ', acc_scores)
    print('avg acc_scores ', np.mean(acc_scores))
    print('std acc_scores ', np.std(acc_scores))
    print('pre_scores ', pre_scores)
    print('avg pre_scores ', np.mean(pre_scores))
    print('std pre_scores ', np.std(pre_scores))
    print("recall_scores", recall_scores)
    print('avg recall_scores ', np.mean(recall_scores))
    print('std recall_scores ', np.std(recall_scores))

    return
