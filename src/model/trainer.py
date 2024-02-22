import os
import time

import pandas as pd
import torch
from sklearn import metrics
from sklearn.metrics import roc_auc_score

from configs.config import config_model
from src.init import logger
from src.utils.utils import calculate


class Trainer:
    def __init__(self, model, device, optimizer, criterion, model_name):
        self.model = model
        self.device = device
        self.optimizer = optimizer
        self.criterion = criterion

        self.best_valid_score = 0
        self.n_patience = 0
        self.last_model = None
        self.model_name = model_name

    def fit(self, epochs, train_loader, valid_loader, patience, fold):
        best_auc = 0
        for n_epoch in range(1, epochs + 1):
            logger.info(f'Epoch {n_epoch}')

            train_loss, train_auc, train_time, rst_train = self.train_epoch(train_loader)
            valid_loss, valid_auc, valid_time, rst_valid = self.val_epoch(valid_loader)

            logger.info(
                f'Epoch Train: {n_epoch}, Loss: {train_loss:.4f}, auc: {train_auc:.4f}, time: {train_time:.2f} s')
            logger.info(
                f'Epoch Valid: {n_epoch}, Loss: {valid_loss:.4f}, auc: {valid_auc:.4f}, time: {valid_time:.2f} s')

            if self.best_valid_score < valid_auc and n_epoch > 5:
                #             if self.best_valid_score > valid_loss:
                self.save_model(n_epoch, config_model[self.model_name], fold)
                logger.info(
                    f'Auc increase from {self.best_valid_score:.4f} to {valid_auc:.4f}. save model to {self.last_model}')

                self.best_valid_score = valid_auc
                self.n_patience = 0
                final_rst_train = rst_train
                final_rst_val = rst_valid
            else:
                self.n_patience += 1

            if self.n_patience >= patience:
                logger.info(f'Valid acc did not improve after {patience}')
                break

        all_rst = [final_rst_train, final_rst_val]
        rst = pd.concat(all_rst, axis=1)
        logger.info(f'rst: {rst}')

        logger.info(f'fold {fold} finished!')
        return rst

    def train_epoch(self, train_loader):
        self.model.train()

        t = time.time()
        sum_loss = 0
        labels = []
        predictions = []

        for step, batch in enumerate(train_loader, 1):
            # 将输入数据和目标标签移动到所选择的设备（GPU或CPU）
            X = batch[0].to(self.device)
            targets = batch[1].to(self.device)
            targets = targets.float()
            # 梯度置零
            self.optimizer.zero_grad()
            # 前向传播
            outputs = self.model(X).squeeze(1)
            # print(targets, targets.shape)
            # print(outputs, outputs.shape)

            # 计算损失并进行反向传播
            loss = self.criterion(outputs, targets)
            loss.backward()

            # 累加总损失
            sum_loss += loss.detach().item()

            labels.extend(batch[1].tolist())
            predictions.extend(outputs.tolist())

            self.optimizer.step()
            # logger.info(f'Train Step {step} / {len(train_loader)}, Loss {sum_loss / step}')
            # logger.info(f'labels: {labels}, predictions: {predictions}')

        auc = roc_auc_score(labels, predictions)
        fpr_micro, tpr_micro, th = metrics.roc_curve(labels, predictions)
        max_th = -1
        max_yd = -1
        for i in range(len(th)):
            yd = tpr_micro[i] - fpr_micro[i]
            if yd > max_yd:
                max_yd = yd
                max_th = th[i]

        rst_train, pred = calculate(predictions, labels, max_th)
        rst_train = pd.DataFrame([rst_train])

        # logger.info(f'labels: {labels}, \n'
        #             f'Predictions: {pred.tolist()}')
        return sum_loss / len(train_loader), auc, int(time.time() - t), rst_train

    def val_epoch(self, val_loader):
        self.model.eval()
        t = time.time()
        sum_loss = 0
        labels = []
        predictions = []

        for step, batch in enumerate(val_loader, 1):
            with torch.no_grad():
                X = batch[0].to(self.device)
                targets = batch[1].to(self.device)
                targets = targets.float()

                # 前向传播
                outputs = self.model(X).squeeze(1)

                # 计算损失并进行反向传播
                loss = self.criterion(outputs, targets)

                # 累加总损失
                sum_loss += loss.detach().item()

                labels.extend(batch[1].tolist())
                predictions.extend(outputs.tolist())
                # print(targets, targets.shape)
                # print(outputs, outputs.shape)

                # logger.info(f'Valid Step {step} / {len(val_loader)}, Loss {(sum_loss / step)}')

        auc = roc_auc_score(labels, predictions)
        fpr_micro, tpr_micro, th = metrics.roc_curve(labels, predictions)
        max_th = -1
        max_yd = -1
        for i in range(len(th)):
            yd = tpr_micro[i] - fpr_micro[i]
            if yd > max_yd:
                max_yd = yd
                max_th = th[i]

        # rst_val = pd.DataFrame([calculate(predictions, labels, max_th)])
        rst_val, pred = calculate(predictions, labels, max_th)
        rst_val = pd.DataFrame([rst_val])

        # logger.info(f'labels: {labels}, \n'
        #             f'Predictions: {pred.tolist()}')

        return sum_loss / len(val_loader), auc, int(time.time() - t), rst_val

    def save_model(self, n_epoch, save_path, fold):
        os.makedirs(save_path, exist_ok=True)
        # self.last_model = os.path.join(save_path, f"fold_{fold}.pth")
        self.last_model = save_path + '/' + f'fold_{fold + 1}.pth'

        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "best_valid_score": self.best_valid_score,
                "n_epoch": n_epoch,
            },
            self.last_model,
        )
