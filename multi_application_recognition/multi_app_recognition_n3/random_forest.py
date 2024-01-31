import pickle
import pandas as pd
from matplotlib import pyplot as plt
from mlxtend.plotting import plot_confusion_matrix
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split


def modelTrain(dataset_path, save_model):

    my_traces_timer = pd.read_csv(dataset_path)
    y = my_traces_timer.iloc[:, 0]
    X = my_traces_timer.iloc[:, 1:]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    forest100 = RandomForestClassifier(n_estimators=100, random_state=42)
    forest100.fit(X_train, y_train)
    y_pred = forest100.predict(X_test)

    message = 'testing n_estimators=100\n' + \
              "Accuracy on test set: {:.3f}".format(forest100.score(X_test, y_test)) + '\n' + \
              classification_report(y_test, y_pred) + '\n'

    f = open(save_model, 'wb')
    pickle.dump(forest100, f)
    f.close()

    return message


def modelTest(dataset_path, save_model, label_text, save_pic):
    my_traces_timer = pd.read_csv(dataset_path)
    y = my_traces_timer.iloc[:, 0]
    X = my_traces_timer.iloc[:, 1:]
    forest100 = pickle.load(open(save_model, 'rb'))
    y_pred = forest100.predict(X)

    message = 'testing n_estimators=100\n' + \
              "Accuracy on test set: {:.3f}".format(forest100.score(X, y)) + '\n' + \
              classification_report(y, y_pred) + '\n'

    cm = confusion_matrix(y, y_pred)

    plt.rcParams["font.family"] = "Times New Roman"
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
    plt.savefig(save_pic, dpi=300, bbox_inches='tight')  # , bbox_inches='tight'
    plt.close('all')
    return message
