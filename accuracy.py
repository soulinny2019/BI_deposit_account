from sklearn import metrics


def accuracy(y_test, y_pred):
    print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
