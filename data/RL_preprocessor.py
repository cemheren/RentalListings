import ijson
import json
import numpy as np

interestResolver = dict()
interestResolver['low'] = 2
interestResolver['medium'] = 1
interestResolver['high'] = 0

def find_keep_mask_for_price(price_list, upper_perc=None, lower_perc=None):
    if upper_perc is None:
        upper_perc = 99.95
    if lower_perc is None:
        lower_perc = 0.005
    # First Convert to NP_ARRAY
    price_npv = np.array(price_list)
    # Eval Percentile Limit Values
    llimit = np.percentile(price_npv, lower_perc)
    ulimit = np.percentile(price_npv, upper_perc)
    print '###############################################'
    print 'Preprocessing (Find Outlier Indices): Price '
    print 'llimit=', llimit, '\t\tulimit=', ulimit
    print '###############################################'
    price_keep_mask = (price_npv < ulimit) & (price_npv > llimit)
    # This price_keep_mask is a boolean list for the entries to be kept
    return price_keep_mask


def normalize_col(input_matrix, col_idx):
    num_cols = input_matrix.shape.__len__()
    # Eval Mean Stdev
    if num_cols == 1:
        col_mean = np.mean(input_matrix)
        col_std = np.std(input_matrix)
        normalized_matrix = (input_matrix - col_mean) / col_std
    else:
        col_mean = np.mean(input_matrix[:, col_idx])
        col_std = np.std(input_matrix[:, col_idx])
        normalized_matrix = input_matrix
        normalized_matrix[:, col_idx] = (input_matrix[:, col_idx] - col_mean) / col_std
    print '###############################################'
    print 'Preprocessing (Normalization): col_idx=', col_idx
    print 'Mean=', col_mean, '\t\t Stdev=', col_std
    print '###############################################'
    return normalized_matrix
