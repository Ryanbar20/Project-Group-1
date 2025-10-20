# coding=utf-8
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Utility functions for S-GAIN:

(1) binary_sampler: sample binary random variables
(2) uniform_sampler: sample uniform random variables
(3) sample_batch_index: sample index of the mini-batch
(3) normalization: normalize the data in [0, 1] range
(4) renormalization: re-normalize data from [0, 1] range to the original range
(5) rounding: round the imputed data for categorical variables
(6) mar_sampler: sample mar-based random variables
"""

import numpy as np


def binary_sampler(p, rows, cols, seed=None):
    """Sample binary random variables.

    :param p: the probability of 1
    :param rows: the number of rows
    :param cols: the number of columns
    :param seed: the random seed

    :return:
    - binary_random_matrix: a binary random matrix
    """

    # Fix seed for run-to-run consistency
    if seed is not None: np.random.seed(seed)

    uniform_random_matrix = np.random.uniform(0., 1., size=(rows, cols))
    binary_random_matrix = 1 * (uniform_random_matrix < p)
    return binary_random_matrix


def uniform_sampler(low, high, rows, cols):
    """Sample uniform random variables.

    :param low: the low limit
    :param high: the high limit
    :param rows: the number of rows
    :param cols: the number of columns

    :return:
    - uniform_random_matrix: a uniform random matrix
    """

    uniform_random_matrix = np.random.uniform(low, high, size=(rows, cols))
    return uniform_random_matrix


def sample_batch_index(total, batch_size):
    """Sample index of the mini-batch.

    :param total: the total number of samples
    :param batch_size: the batch size

    Returns:
    - batch_idx: the batch index
    """

    total_idx = np.random.permutation(total)
    batch_idx = total_idx[:batch_size]
    return batch_idx


def normalization(data_x, norm_parameters=None):
    """Normalize the data in [0, 1] range.

    :param data_x: the original data

    :return:
    - norm_data_x: normalized data
    - norm_parameters: min_val, max_val for each feature for renormalization
    """

    # Parameters
    _, dim = data_x.shape
    norm_data_x = data_x.copy()

    if norm_parameters is None:
        min_val = np.zeros(dim)
        max_val = np.zeros(dim)

        for i in range(dim):  # Todo: run on GPU?
            min_val[i] = np.nanmin(norm_data_x[:, i])
            norm_data_x[:, i] = norm_data_x[:, i] - np.nanmin(norm_data_x[:, i])
            max_val[i] = np.nanmax(norm_data_x[:, i])
            norm_data_x[:, i] = norm_data_x[:, i] / (np.nanmax(norm_data_x[:, i]) + 1e-7)

        norm_parameters = {'min_val': min_val, 'max_val': max_val}

    else:
        min_val = norm_parameters['min_val']
        max_val = norm_parameters['max_val']

        for i in range(dim):  # Todo: run on GPU?
            norm_data_x[:, i] = norm_data_x[:, i] - min_val[i]
            norm_data_x[:, i] = norm_data_x[:, i] / (max_val[i] + 1e-7)

    return norm_data_x, norm_parameters


def renormalization(norm_data_x, norm_parameters):
    """Re-normalize data from [0, 1] range to the original range.

    :param norm_data_x: the normalized data
    :param norm_parameters: the min_val and max_val for each feature for renormalization

    :returns:
    - renorm_data_x: the re-normalized data
    """

    min_val = norm_parameters['min_val']
    max_val = norm_parameters['max_val']

    _, dim = norm_data_x.shape
    renorm_data_x = norm_data_x.copy()

    for i in range(dim):  # Todo: run on GPU?
        renorm_data_x[:, i] = renorm_data_x[:, i] * (max_val[i] + 1e-7)
        renorm_data_x[:, i] = renorm_data_x[:, i] + min_val[i]

    return renorm_data_x


def rounding(imputed_data_x, miss_data_x):
    """Round the imputed data for categorical variables.

    :param imputed_data_x: the imputed data
    :param miss_data_x: the data with missing values

    Returns:
    - rounded_data_x: the rounded data
    """

    _, dim = miss_data_x.shape
    rounded_data_x = imputed_data_x.copy()

    for i in range(dim):  # Todo: run on GPU?
        temp = miss_data_x[~np.isnan(miss_data_x[:, i]), i]

        # Only for the categorical variables
        if len(np.unique(temp)) < 20:
            rounded_data_x[:, i] = np.round(rounded_data_x[:, i])

    return rounded_data_x


def mar_sampler(p, no, dim, x, seed=None):
    """Sample MAR distributed random variables

        This method generates a mask of binary values for the missing data. 
        This is done according to the formula from the paper from Yoon et al.
        The Missing data is dependent on occurences of previous data points.

        
        :param p    : average probability of a datapoint missing 
        :param no   : amount of records
        :param dim  : amount of attributes
        :param x    : the original dataset
        :param seed : None is the base value, set this to make the random sampling deterministic

        :return     : mask on the data showing the missings values (0) and the observed ones (1)
    """
    if seed: np.random.seed(seed)

    #The matrix of the w and b values for every j, the first value is w and second is b
    wb_matrix = np.random.uniform(0.,1.,size=(2,dim)) 
    if seed: np.random.seed(seed)
    #init mask array
    mask = np.random.uniform(0., 1., size=(no,dim ))

    #set average missing rate of i-th feature equal for all the features
    
    for i in range(dim):
        vectors = []

        for l in range(no):
            vec = wb_matrix[0] * mask[l] * x[l] + wb_matrix[1]*(1-mask[l])
            vectors.append(np.sum(vec[0:i]))
        vectors = np.array(vectors)
        for n in range(no):
            divisor = np.sum(np.exp(-vectors))
            result = p * no * np.exp(-vectors[n]) / divisor
            mask[n][i] = 1 * (mask[n][i] < result)
    mask = 1 - mask
    print(mask[:5])
    return mask

def gain_mnar_sampler(p, no, dim, x, seed=None):
    """ Sample MNAR distributed random variables

    p = probability
    N = number of examples
    w = weights sampled U(0,1)
    x = datapoints
    """

    if seed: np.random.seed(seed)

    #The matrix of the w and b values for every j, the first value is w and second is b
    w_array = np.random.uniform(0.,1.,size=(dim,)) 

    #init mask array
    mask = np.random.uniform(0., 1., size=(no,dim ))

    denominators = np.zeros(dim)
    for i in range(dim):
        for j in range(no):
            denominators[i] += np.exp(-w_array[i] * x[j][i])
    
    for i in range(dim):
        for j in range(no):
            nominator = p * no * np.exp(-w_array[i] * x[j][i])
            mask[j][i] = 1 * (mask[j][i] < (nominator/denominators[i]))
    return 1 - mask