import numpy as np
from sklearn import metrics


def calculate(score, label, th):
    score = np.array(score)
    label = np.array(label)
    pred = np.zeros_like(label)
    pred[score >= th] = 1
    pred[score < th] = 0

    TP = len(pred[(pred > 0.5) & (label > 0.5)])
    FN = len(pred[(pred < 0.5) & (label > 0.5)])
    TN = len(pred[(pred < 0.5) & (label < 0.5)])
    FP = len(pred[(pred > 0.5) & (label < 0.5)])

    AUC = metrics.roc_auc_score(label, score)
    result = {'AUC': AUC, 'acc': (TP + TN) / (TP + TN + FP + FN), 'rec': (TP) / (TP + FN + 0.0001),
              'spe': (TN) / (TN + FP + 0.0001), 'pre': (TP) / (TP + FP + 0.0001), 'TP': TP, 'TN': TN, 'FP': FP,
              'FN': FN}
    #     print('acc',(TP+TN),(TP+TN+FP+FN),'spe',(TN),(TN+FP),'sen',(TP),(TP+FN))
    return result, pred
