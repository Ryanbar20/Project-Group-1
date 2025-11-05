"""Dataset loader for S-GAIN:

(1) data_loader: load a dataset and introduce missing elements
"""

import numpy as np
from PIL import Image
from utils.utils import binary_sampler, mar_sampler, mnar_sampler, upscale
from keras.datasets import mnist, fashion_mnist, cifar10


def data_loader(dataset, miss_rate, miss_modality, seed=None):
    """Load a dataset and introduce missing elements.

    Todo: other miss modalities [MAR, MNAR, AI_upscaler, square]

    :param dataset: the dataset to use
    :param miss_rate: the probability of missing elements in the data
    :param miss_modality: the modality of missing data [MCAR, MAR, MNAR, AI_upscaler, square]
    :param seed: the seed used to introduce missing elements in the data

    :return:
    - data_x: the original data (without missing values)
    - miss_data_x: the data with missing values
    - data_mask: the indicator matrix for missing elements
    """
    data_points_per_pixel = 1
    # Load the data
    if dataset in ['health', 'letter', 'spam']:
        file_name = f'datasets/{dataset}.csv'
        data_x = np.loadtxt(file_name, delimiter=',', skiprows=1)
    elif dataset == 'mnist':
        (data_x, _), _ = mnist.load_data()
        data_x = np.reshape(np.asarray(data_x), [60000, 28 * 28]).astype(float)
    elif dataset == 'fashion_mnist':
        (data_x, _), _ = fashion_mnist.load_data()
        data_x = np.reshape(np.asarray(data_x), [60000, 28 * 28]).astype(float)
    elif dataset == 'cifar10':
        (data_x, _), _ = cifar10.load_data()
        data_points_per_pixel = 3
        data_x = np.reshape(np.asarray(data_x), [50000, 32 * 32 * 3]).astype(float)
    elif dataset == 'test':
        data_x =  Image.open(f'datasets/{dataset}.jpg')
        data_x = data_x.convert('L')
        data_x = np.array(data_x).astype(float)
        print(data_x[:5])
        if (len(data_x.shape) == 3):
            data_points_per_pixel = data_x.shape[2]
            data_x = data_x.reshape((data_x.shape[0], data_x.shape[1] * data_x.shape[2]))


        
    else:  # This should not happen
        print(f'Invalid dataset: "{dataset}". Exiting the program.')
        return None

    # Introduce missing elements in the data'
    no, dim = data_x.shape[:2]
    match miss_modality:
        case 'MCAR':
            data_mask = binary_sampler(1 - miss_rate, no, dim, seed)
        case 'MAR':
            data_mask = mar_sampler(miss_rate, no, dim, data_x, seed)
        case 'MNAR':
            data_mask = mnar_sampler(miss_rate,no,dim,data_x,seed)
        case 'AI_UPSCALER':
            data_mask = upscale(data_x, miss_rate, data_points_per_pixel)
    print(data_mask[:5])
    miss_data_x = data_x.copy()
    miss_data_x[data_mask == 0] = np.nan
    return data_x, miss_data_x, data_mask
