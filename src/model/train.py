import time

import pandas as pd
import torch
from torch import optim, nn

from configs.config import config_train, config_result
from src.init import dataloaders, device
from src.init import logger
from src.model.ResNet import ResNet18
from src.model.denseNet import DenseNet
from src.model.Unet import Unet
from src.model.trainer import Trainer


# from src.model.Unet import UNet, UNetWithClassifier


def train(model, train_loader, val_loader, criterion, optimizer, device, num_epochs=5):
    # logger = getLogger(config_logger['train'])
    model = model.to(device)
    criterion = criterion.to(device)

    train_losses = []
    val_accuracies = []

    for epoch in range(num_epochs):

        model.train()  # 将模型设置为训练模式
        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data[0].to(device), data[1].to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 50 == 49:  # 每 50 个 mini-batch 打印一次损失
                logger.info('[%d, %5d] loss: %.5f' %
                            (epoch + 1, i + 1, running_loss / 100))
                running_loss = 0.0

        train_losses.append(running_loss / len(train_loader))

        # 在验证集上评估模型
        accuracy = validate(val_loader, model, device)
        val_accuracies.append(accuracy)
        logger.info('Epoch {} - Validation Accuracy: {:.2f}%'.format(epoch + 1, accuracy))

    return train_losses, val_accuracies, model


def validate(loader, model, device):
    correct = 0
    total = 0
    model.eval()  # 将模型设置为评估模式
    with torch.no_grad():
        for data in loader:
            inputs, labels = data[0].to(device), data[1].to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            # print(f'predicted:{predicted}')
            # print(f'labels:{labels}')
            total += labels.size(0)
            print(predicted)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total

    return accuracy


if __name__ == '__main__':
    logger.info(f'Using device: {device}')

    rst_dfs = []

    for fold, (train_loader, val_loader) in enumerate(dataloaders):
        logger.info(f'Training for fold {fold + 1} ...')

        # model = ResNet18(device)
        # model = Unet(3, 1)
        model = DenseNet()
        model = model.to(device)
        model_name = 'DenseNet'

        optimizer = optim.Adam(model.parameters(), lr=config_train['learning_rate'])
        # criterion = nn.CrossEntropyLoss()
        # criterion = nn.BCEWithLogitsLoss()
        criterion = nn.BCELoss()

        trainer = Trainer(model, device, optimizer, criterion, model_name)
        rst = trainer.fit(config_train['epochs'], train_loader, val_loader, config_train['patience'], fold)
        rst_dfs.append(rst)

    rst_dfs = pd.concat(rst_dfs)
    logger.info(rst_dfs)

    rst_dfs = pd.DataFrame(rst_dfs)
    rst_dfs.to_csv(config_result[model_name] + '/train_val_res_pf_' + str(time.time()) + '.csv')
