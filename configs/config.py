config_train = {
    'batch_size': 8,
    'epochs': 15,
    'learning_rate': 1e-4,
    'k-fold': 5,
    'patience': 30,

}
root = 'F:/Coursework/Final year project/AD classification'

config_data = {
    'root': root,
    'raw': 'H:/raw',
    'pMCI': '/pMCI_caps/subjects',
    'sMCI': '/sMCI_caps/subjects',
    'subject': '/ses-M000/deeplearning_prepare_data',
    'slice': '/slice_based/custom',
    'pMCI_sag': '/dataset/processed/pMCI_sag',
    'sMCI_sag': root + '/dataset/processed/sMCI_sag',
    'pMCI_axi': root + '/dataset/processed/pMCI_axi',
    'sMCI_axi': root + '/dataset/processed/sMCI_axi',
    'start': 80,
    'length': 5,
}

config_logger = {
    'main': config_data['root'] + '/src',
}

config_model = {
    'UNet': root + '/models/UNet',
    'ResNet': root + '/models/ResNet',
    'DenseNet': root + '/models/DenseNet',
}

config_result = {
    'ResNet': root + '/results/ResNet',
    'UNet': root + '/results/UNet',
    'DenseNet': root + '/results/DenseNet',
}
