import torch

from configs.config import config_logger
from configs.config import config_train
from log.log import getLogger
from src.data_processing.dataloader import get_kfold_dataloader

logger = getLogger(config_logger['main'])
dataloaders, test_dataloader = get_kfold_dataloader(k=config_train['k-fold'])
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
