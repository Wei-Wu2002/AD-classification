from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, ConcatDataset

from configs.config import config_data, config_train
from src.data_processing.dataset import CustomDataset


def make_dataset(pMCI_path, sMCI_path):
    pMCI_dataset = CustomDataset(config_data['pMCI_axi'] + '/' + pMCI_path[0], label=1)
    # print(pMCI_dataset.label)
    for subject in pMCI_path[1:]:
        pMCI_dataset = ConcatDataset([pMCI_dataset, CustomDataset(config_data['pMCI_axi'] + '/' + subject, label=1)])
        # print(pMCI_dataset.datasets.file_list)
        # print(temp_dataset.label)

    sMCI_dataset = CustomDataset(config_data['sMCI_axi'] + '/' + sMCI_path[0], label=0)
    # print(pMCI_dataset.label)
    for subject in sMCI_path[1:]:
        sMCI_dataset = ConcatDataset([sMCI_dataset, CustomDataset(config_data['sMCI_axi'] + '/' + subject, label=0)])
        # print(temp_dataset.file_list)
        # print(temp_dataset.label)

    # print(pMCI_dataset.file_list, pMCI_dataset.label)

    # sMCI_dataset = CustomDataset(config_data['sMCI_data'] + '/' + sMCI_path[0], label=0)
    # # print(sMCI_dataset.label)
    # for subject in sMCI_path[1:]:
    #     temp_dataset = CustomDataset(config_data['sMCI_data'] + '/' + subject, label=0)
    #     # print(temp_dataset.file_list)
    #     # print(temp_dataset.label)
    #     sMCI_dataset.add_sample(temp_dataset)
    dataset = ConcatDataset([pMCI_dataset, sMCI_dataset])

    return dataset


def get_dataloader():
    # 加载数据集
    pmci_dataset = CustomDataset(config_data['pMCI_axi'], label=1)
    smci_dataset = CustomDataset(config_data['sMCI_axi'], label=0)

    pmci_train_val_subjects, pmci_test_subjects = train_test_split(pmci_dataset.file_list, test_size=0.1,
                                                                   random_state=42)
    smci_train_val_subjects, smci_test_subjects = train_test_split(smci_dataset.file_list, test_size=0.1,
                                                                   random_state=42)
    pmci_train_subjects, pmci_val_subjects = train_test_split(pmci_train_val_subjects, test_size=0.1, random_state=42)
    smci_train_subjects, smci_val_subjects = train_test_split(smci_train_val_subjects, test_size=0.1, random_state=42)

    # print(f'pmci_train_subjects: {len(pmci_train_subjects)} {pmci_train_subjects},\n'
    #       f'pmci_val_subjects:{len(pmci_val_subjects)} {pmci_val_subjects},\n'
    #       f'pmci_test_subjects:{len(pmci_test_subjects)} {pmci_test_subjects}')

    train_dataset = make_dataset(pmci_train_subjects, smci_train_subjects)
    val_dataset = make_dataset(pmci_val_subjects, smci_val_subjects)
    test_dataset = make_dataset(pmci_test_subjects, smci_test_subjects)

    # 创建数据加载器
    train_dataloader = DataLoader(train_dataset, batch_size=config_train['batch_size'], shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=config_train['batch_size'], shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=config_train['batch_size'], shuffle=True)

    return train_dataloader, val_dataloader, test_dataloader


from sklearn.model_selection import KFold


def get_kfold_dataloader(k=5):
    # 加载数据集
    pmci_dataset = CustomDataset(config_data['pMCI_axi'], label=1)
    smci_dataset = CustomDataset(config_data['sMCI_axi'], label=0)

    pmci_train_val_subjects, pmci_test_subjects = train_test_split(pmci_dataset.file_list, test_size=0.1,
                                                                   random_state=42)
    smci_train_val_subjects, smci_test_subjects = train_test_split(smci_dataset.file_list, test_size=0.1,
                                                                   random_state=42)
    # pmci_train_subjects, pmci_val_subjects = train_test_split(pmci_train_val_subjects, test_size=0.25, random_state=42)
    # smci_train_subjects, smci_val_subjects = train_test_split(smci_train_val_subjects, test_size=0.25, random_state=42)

    dataloaders = []
    train_subjects = pmci_train_val_subjects + smci_train_val_subjects

    # 划分数据集并创建数据加载器
    for fold, (train_index, val_index) in enumerate(
            KFold(n_splits=k, shuffle=True, random_state=42).split(pmci_train_val_subjects)):
        train_dataset = make_dataset([pmci_train_val_subjects[i] for i in train_index],
                                     [smci_train_val_subjects[i] for i in train_index])
        val_dataset = make_dataset([pmci_train_val_subjects[i] for i in val_index],
                                   [smci_train_val_subjects[i] for i in val_index])

        train_dataloader = DataLoader(train_dataset, batch_size=config_train['batch_size'], shuffle=True)
        val_dataloader = DataLoader(val_dataset, batch_size=config_train['batch_size'], shuffle=True)

        dataloaders.append((train_dataloader, val_dataloader))

    test_dataset = make_dataset(pmci_test_subjects, smci_test_subjects)
    test_dataloader = DataLoader(test_dataset, batch_size=config_train['batch_size'], shuffle=True)

    return dataloaders, test_dataloader


# 在这里调用 get_kfold_dataloader() 函数即可获取 k 折交叉验证的数据加载器


if __name__ == '__main__':
    # train_dataloader, val_dataloader, test_dataloader = get_dataloader()
    # # 测试训练集的DataLoader
    # for batch_idx, (data, labels) in enumerate(test_dataloader):
    #     print(f"Batch {batch_idx}:")
    #     print("Data shape:", data.shape)
    #     print("Labels:", labels)
    # 调用 get_kfold_dataloader() 函数获取数据加载器
    dataloaders, test_dataloader = get_kfold_dataloader(k=config_train['k-fold'])

    for batch_idx, (data, labels) in enumerate(test_dataloader):
        print(f"Batch {batch_idx}:")
        print("Data shape:", data.shape)
        print("Labels:", labels)

    # 遍历生成的数据加载器，并检查每个加载器中的样本数量等信息
    for fold, (train_loader, val_loader) in enumerate(dataloaders):
        print(f"Fold {fold + 1}:")
        print(f"Number of training samples: {len(train_loader.dataset)}")
        print(f"Number of validation samples: {len(val_loader.dataset)}")
        print("Sample batch shapes:")
        for batch_idx, (data, labels) in enumerate(train_loader):
            print(f"  Batch {batch_idx}: Data shape - {data.shape}, Labels - {labels}")
        print("--------------------------------------------------------")
        for batch_idx, (data, labels) in enumerate(val_loader):
            print(f"  Batch {batch_idx}: Data shape - {data.shape}, Labels - {labels}")
        print("--------------------------------------------------------")
