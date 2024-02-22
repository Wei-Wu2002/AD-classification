import os
import time

import pandas as pd
import torch
from sklearn import metrics

from configs.config import config_result, config_model
from src.init import logger
from src.init import test_dataloader, device
from src.model.ResNet import ResNet18
from src.model.denseNet import DenseNet
from src.model.Unet import Unet
from src.utils.utils import calculate


def predict(model_file):

    # model = Unet(3, 1)
    # model = DenseNet()
    model = ResNet18(device)
    model.to(device)

    checkpoint = torch.load(model_file)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    lables = []
    predictions = []

    for epoch, batch in enumerate(test_dataloader, 1):
        logger.info(f'Epoch {epoch}/{len(test_dataloader)}')
        with torch.no_grad():
            temp_pred = model(batch[0].to(device)).cpu().numpy().squeeze()
            targets = batch[1].to(device)

            if temp_pred.size == 1:
                predictions.append(temp_pred)
            else:
                predictions.extend(temp_pred.tolist())
            lables.extend(batch[1].tolist())

    labels = [1 if x > 0.5 else 0 for x in lables]
    fpr_micro, tpr_micro, th = metrics.roc_curve(labels, predictions)
    max_th = -1
    max_yd = -1
    for i in range(len(th)):
        yd = tpr_micro[i] - fpr_micro[i]
        if yd > max_yd:
            max_yd = yd
            max_th = th[i]

    rst_test, pred = calculate(predictions, labels, max_th)
    rst_test = pd.DataFrame([rst_test])

    pred_df = pd.DataFrame({'label': lables, 'prediction': pred.tolist()})
    return pred_df, rst_test

if __name__ == '__main__':
    model_name = 'ResNet'
    model_files = os.listdir(config_model[model_name])
    logger.info(f'Model files: {model_files}')


    rst_tests = []
    df_test = {}

    for model_file in model_files:
        pred_df, rst_test = predict(config_model[model_name] + '/' + model_file)
        rst_tests.append(rst_test)
        df_test['prediction'] = pred_df['prediction']

    rst_tests = pd.concat(rst_tests)
    rst_tests = pd.DataFrame(rst_tests)

    df_test['label'] = pred_df['label']
    df_test = pd.DataFrame(df_test)

    rst_tests.loc['mean'] = rst_tests.mean(axis=0)
    rst_tests.to_csv(config_result[model_name] + '/rst_test_' + str(time.time()) +'.csv')
    logger.info(f'Mean: {rst_tests}')


