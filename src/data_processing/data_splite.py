import os
import re
import shutil

from configs.config import config_data
from src.init import logger


def get_files_in_dir(path, items, type):
    pattern = r"slice-(\d+)_"
    template = r"axis-axi"
    if type == 'pMCI':
        destination_root = config_data['pMCI_axi']
    elif type == 'sMCI':
        destination_root = config_data['sMCI_axi']
    else:
        raise Exception('Type error')

    for item in items:
        # 按subject划分
        destination_folder = destination_root + '/' + item

        # destination_folder = destination_root
        if not os.path.exists(destination_folder):
            os.makedirs(destination_folder)
        file_path = path + '/' + item + config_data['subject'] + config_data['slice']
        tensor_list = os.listdir(file_path)
        # logger.info(f'file_path: {len(tensor_list)}')

        for tensor in tensor_list:
            match = re.search(pattern, tensor)
            if match and re.search(template, tensor):

                slice_number = int(match.group(1))
                if config_data['start'] <= slice_number < config_data['start'] + config_data['length']:
                    tensor_path = file_path + '/' + tensor
                    logger.info(tensor_path)
                    shutil.copy(tensor_path, destination_folder)


if __name__ == '__main__':

    pMCI_path = config_data['raw'] + config_data['pMCI']
    sMCI_path = config_data['raw'] + config_data['sMCI']

    items = os.listdir(pMCI_path)
    get_files_in_dir(pMCI_path, items, 'pMCI')

    items = os.listdir(sMCI_path)
    get_files_in_dir(sMCI_path, items, 'sMCI')