import torch
from torch import optim, nn

from src.init import logger
from data_processing.dataloader import get_dataloader
from src.model.ResNet import ResNet18
# from src.model.Unet import UNet, UNetWithClassifier
from src.model.Unet import Unet
from src.model.train import train, validate
from configs.config import config_train

if __name__ == '__main__':
    # device = 'cpu'
    device = torch.device("cuda" if torch.cuda.is_available() else "")
    logger.info(f'Using device: {device}')

    train_dataloader, val_dataloader, test_dataloader = get_dataloader()
    model = ResNet18(device)
    # model = UNetWithClassifier(3, 1, 2)
    # model = Unet(3, 2)
    criterion = nn.CrossEntropyLoss()
    # criterion = nn.BCELoss()
    # criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    # optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    train_losses, val_accuracies, model_trained = train(model, train_dataloader, val_dataloader, criterion, optimizer, device, num_epochs= config_train['epochs'])

    test_accuracies = validate(test_dataloader, model_trained, device)
    logger.info(f'test_accuracies: {test_accuracies}')

