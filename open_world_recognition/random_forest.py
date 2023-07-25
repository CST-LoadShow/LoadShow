import pickle
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
from mlxtend.plotting import plot_confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier


def modelTrain(dataset_path_0, dataset_path_1, test_path, test_size_time=10):
    my_traces_timer_0 = pd.read_csv(dataset_path_0)
    y_0 = my_traces_timer_0.iloc[:, 0]
    X_0 = my_traces_timer_0.iloc[:, 1:]
    X_train_0, X_test_0, y_train_0, y_test_0 = train_test_split(X_0, y_0, test_size=0.2, random_state=42)

    my_traces_timer_1 = pd.read_csv(dataset_path_1)
    y_1 = my_traces_timer_1.iloc[:, 0]
    X_1 = my_traces_timer_1.iloc[:, 1:]
    X_train_1, X_test_1, y_train_1, y_test_1 = train_test_split(X_1, y_1, test_size=0.2, random_state=42)

    my_traces_timer = pd.read_csv(test_path)
    y_2 = my_traces_timer.iloc[:, 0]
    X_2 = my_traces_timer.iloc[:, 1:]
    X_train_2, X_test_2, y_train_2, y_test_2 = train_test_split(X_2, y_2, test_size=0.2/15 * test_size_time, random_state=42)

    X_train = pd.concat([X_train_0, X_train_1])
    y_train = pd.concat([y_train_0, y_train_1])

    X_test = pd.concat([X_test_0, X_test_2])
    y_test = pd.concat([y_test_0, y_test_2])
    forest100 = RandomForestClassifier(n_estimators=100, random_state=0)
    forest100.fit(X_train, y_train)
    y_pred = forest100.predict(X_test)

    message = "n_estimators=100\n" + \
              "Accuracy on training set: {:.3f}\n".format(forest100.score(X_train, y_train)) + \
              "Accuracy on val set: {:.3f}\n".format(forest100.score(X_test, y_test)) + \
              classification_report(y_test, y_pred)
    dic = classification_report(y_test, y_pred, output_dict=True)
    cm = confusion_matrix(y_test, y_pred)
    (TP, FN), (FP, TN) = cm
    FPR = FP / (FP + TN)
    TPR = TP / (TP + FN)
    precision_mine = TP / (TP + FP)
    recall_mine = TP / (TP + FN)

    mine_precision = precision_mine
    mine_recall = recall_mine
    mine_f1 = 2 * precision_mine * recall_mine / (precision_mine + recall_mine)

    return message, [FPR, TPR, mine_precision, mine_recall, mine_f1]


def modelTest(test_dataset_file, save_model, save_pic, label_text):
    f = open(save_model, 'rb')
    forest100 = pickle.load(f)
    f.close()
    my_traces_timer = pd.read_csv(test_dataset_file)

    y_test = my_traces_timer.iloc[:, 0]
    X_test = my_traces_timer.iloc[:, 1:]
    y_pred = forest100.predict(X_test)

    probility = forest100.predict_proba(X_test)
    print("probility", probility)

    message = 'testing n_estimators=100\n' + \
              "Accuracy on test set: {:.3f}".format(forest100.score(X_test, y_test)) + '\n' + \
              classification_report(y_test, y_pred) + '\n'
    print("message")
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plot_confusion_matrix(conf_mat=cm,
                                    show_absolute=True,
                                    show_normed=False,
                                    colorbar=False,
                                    class_names=label_text,
                                    figsize=(10, 10),
                                    cmap='Blues',
                                    )

    plt.xlabel("Predicted Label", fontsize=14)
    plt.ylabel("True Label", fontsize=14)
    plt.savefig(save_pic, dpi=300, bbox_inches='tight')
    plt.close('all')

    return message

