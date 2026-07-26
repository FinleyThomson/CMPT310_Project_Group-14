import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np
import pandas as pd
from Data.Code import SyntheticData as sd



def syntheticKFoldCrossValidation(data, feature_cols, synthetic_info_cols, model, k, n):
    """Performs K-fold cross validation on a given model (executed based on the model_inputs), creating a set of synthetic data for each test set to ensure no data leakage.

        Parameters
        ----------
        data: (pandas data frame) data frame containing all data, used to synthesize data
        feature_cols: (list) list containing column labels for classifying
        synthetic_info_cols: (list) list containing column labels that will be used to synthesize new data, must contain feature_columns as a subset
        model: (class) the model to be evaluated using K-fold CV
        k: (int) k value
        n: (int) number of synthetic datapoints

        Returns
        -------
        confusion_matrix: (nparray) confusion matrix from test results
        
    
    """
    real_data = data[synthetic_info_cols]

    real_data = real_data.sample(frac=1).reset_index(drop=True)

    length = len(real_data)

    div = length // k

    split_real_data = [group for _, group in real_data.groupby(real_data.index // div)]

    if len(split_real_data[-1]) < len(split_real_data[-2]):
        split_real_data[-2] = pd.concat([split_real_data[-2],split_real_data[-1]],ignore_index=True)
        split_real_data.pop()

    real_split = [df[synthetic_info_cols] for df in split_real_data]

    #confusion_matrix_training = np.empty((3,3))
    confusion_matrix = np.zeros((3,3))
    training_accuracy = 0
    training_F1 = 0
    test_accuracy = 0
    test_F1 = 0

    for i in range(k):

        print(i) 

        X_test = real_split[i][feature_cols].values
        y_test = real_split[i].iloc[:,-1].values

        training_folds = real_split[:i] + real_split[i+1:]

        train_real = pd.concat(training_folds, ignore_index=True)

        synth_data = sd.multivariateLognormalDistributionGeneration(train_real, n)

        X_train = pd.concat([train_real[feature_cols], synth_data[feature_cols]], ignore_index=True)[feature_cols].values
        y_train = pd.concat([train_real.iloc[:,-1], synth_data.iloc[:,-1]], ignore_index=True).values

        model.fit(X_train, y_train)

        training_preds = model.predict(X_train)

        #confusion_matrix_training += getConfusionMatrix(training_preds, y_train)

        training_accuracy += accuracy(training_preds, y_train)

        training_F1 += macroF1Score(training_preds, y_train, 3)

        test_preds = model.predict(X_test)

        confusion_matrix += getConfusionMatrix(test_preds, y_test, 3)

        test_F1 += macroF1Score(test_preds, y_test, 3)

        test_accuracy += accuracy(test_preds, y_test)

    training_accuracy *= (1/k) 
    training_F1 *= (1/k) 
    test_accuracy *= (1/k)
    test_F1 *= (1/k)

    return confusion_matrix, training_accuracy, training_F1, test_accuracy, test_F1


def getConfusionMatrix(predictions, actuals, n):
    """Entry i,j counts number of times predictions = i and actuals = j for confusion matrix of n x n"""

    confusion_matrix = np.array([[np.sum((predictions == i) & (actuals == j)) for j in range(n)] for i in range(n)])

    return confusion_matrix


def accuracy(predictions, actuals):

    TP = np.sum(predictions == actuals)

    accuracy = TP / len(actuals)

    return accuracy


def macroF1Score(predictions, actuals, num_classes):

    TP_i = 0
    FP_i = 0
    FN_i = 0

    F1_score = 0

    for i in range(num_classes):
        TP_i = np.sum((predictions == i) & (actuals == i))
        FP_i = np.sum((predictions == i) & (actuals != i))
        FN_i = np.sum((predictions != i) & (actuals == i))

        denom_prec = (TP_i + FP_i)
        if denom_prec != 0:
            precision_i = TP_i / denom_prec
        else:
            precision_i = 0

        denom_rec = (TP_i + FN_i)
        if denom_rec != 0:
            recall_i = TP_i / denom_rec
        else:
            recall_i = 0

        denom_F1 = (precision_i + recall_i)
        if denom_F1 != 0:
            F1_score += (2*precision_i*recall_i)/denom_F1
        else:
            F1_score = 0

    F1_score = F1_score / num_classes

    return F1_score






















