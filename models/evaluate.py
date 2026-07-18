from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix


def evaluate_model(model, X_test, y_test):

    prediction = model.predict(X_test)

    accuracy = accuracy_score(y_test, prediction)

    report = classification_report(y_test, prediction)

    matrix = confusion_matrix(y_test, prediction)

    return accuracy, report, matrix