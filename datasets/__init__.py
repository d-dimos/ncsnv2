import os
import torch
from datasets.celeba import CelebA
from datasets.ffhq import FFHQ
from torch.utils.data import Subset
import numpy as np

from datasets.dataloaders import MVU_Estimator_Brain
from datasets.utils import get_all_files


def get_dataset(config):
    folder_path = os.path.join(config.project_dir, config.input_dir)
    files = get_all_files(folder_path, pattern='*.h5')

    assert config.data.dataset == 'brain_T2'
    dataset = MVU_Estimator_Brain(files,
                                  input_dir=config.input_dir,
                                  maps_dir=config.maps_dir,
                                  project_dir=config.project_dir,
                                  image_size=(384, 384),
                                  R=4,
                                  pattern='equispaced',
                                  orientation='vertical')

    num_items = len(dataset)
    indices = list(range(num_items))
    random_state = np.random.get_state()
    np.random.seed(2022)
    np.random.shuffle(indices)
    np.random.set_state(random_state)
    train_indices, test_indices = indices[:int(num_items * 0.9)], indices[int(num_items * 0.9):]
    test_dataset = Subset(dataset, test_indices)
    dataset = Subset(dataset, train_indices)

    return dataset, test_dataset


def logit_transform(image, lam=1e-6):
    image = lam + (1 - 2 * lam) * image
    return torch.log(image) - torch.log1p(-image)


def data_transform(config, X):
    if config.data.uniform_dequantization:
        X = X / 256. * 255. + torch.rand_like(X) / 256.
    if config.data.gaussian_dequantization:
        X = X + torch.randn_like(X) * 0.01

    if config.data.rescaled:
        X = 2 * X - 1.
    elif config.data.logit_transform:
        X = logit_transform(X)

    if hasattr(config, 'image_mean'):
        return X - config.image_mean.to(X.device)[None, ...]

    return X


def inverse_data_transform(config, X):
    if hasattr(config, 'image_mean'):
        X = X + config.image_mean.to(X.device)[None, ...]

    if config.data.logit_transform:
        X = torch.sigmoid(X)
    elif config.data.rescaled:
        X = (X + 1.) / 2.

    return torch.clamp(X, 0.0, 1.0)
