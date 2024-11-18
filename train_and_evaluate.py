import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score
import csv
import os

def train_autoencoder(autoencoder, x_train, x_validation):
    """Training autoencoder"""
    autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
    history = autoencoder.fit(x_train, x_train, epochs=20, batch_size=128, validation_data=(x_validation, x_validation))
    return history

def calculate_aupr(y_true, y_scores):
    """Calculate the AUPR value"""
    return average_precision_score(y_true, y_scores)


def StorFile(data, fileName):

    result_folder = 'result'
    if not os.path.exists(result_folder):
        os.makedirs(result_folder)


    file_path = os.path.join(result_folder, fileName)


    with open(file_path, "w", newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(data)

    return

def evaluate_classifier(clf, name, X_test, y_test):
    """Evaluate classifier performance"""
    y_pred = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)[:, 1]
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)
    aupr = calculate_aupr(y_test, y_proba)
    print(f"{name} classifier - Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, F1 Score: {f1}, AUC: {auc}, AUPR: {aupr}")
    return auc, aupr

def TestOutput(classifier, name, X_test, y_test, fold_index):
    """Test the results of the output classifier"""
    ModelTestOutput = classifier.predict_proba(X_test)
    LabelPredictionProb = [[y_test[i], ModelTestOutput[i][1]] for i in range(len(y_test))]
    LabelPrediction = [[y_test[i], 1 if ModelTestOutput[i][1] > 0.5 else 0] for i in range(len(y_test))]

    StorFile(LabelPredictionProb, f"{name}RealAndPredictionProbA+B_fold_{fold_index}.csv")
    StorFile(LabelPrediction, f"{name}RealAndPredictionA+B_fold_{fold_index}.csv")

    aupr = calculate_aupr(y_test, ModelTestOutput[:, 1])
    auc = roc_auc_score(y_test, ModelTestOutput[:, 1])
    print(f"{name} classifier fold {fold_index} - AUC: {auc}, AUPR: {aupr}")
    return aupr, auc
