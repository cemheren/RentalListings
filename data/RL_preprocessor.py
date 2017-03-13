import ijson
import json
import numpy as np


def find_keep_mask_for_price(price_list, upper_perc=None, lower_perc=None):
    if upper_perc is None:
        upper_perc = 99.95
    if lower_perc is None:
        lower_perc = 0.005
    # First Convert or Copy to NP_ARRAY
    price_npv = np.array(price_list)
    # Eval Percentile Limit Values
    llimit = np.percentile(price_npv, lower_perc)
    ulimit = np.percentile(price_npv, upper_perc)
    price_keep_mask = (price_npv < ulimit) & (price_npv > llimit)
    # This price_keep_mask is a boolean list for the entries to be kept
    return price_keep_mask


def normalize_cols(input_matrix_train, input_matrix_test, normalizableColumnResolver, col_name_list):
    for col_name in col_name_list:
        col_idx = normalizableColumnResolver[col_name]
        # Get mean from TRAINING DATA
        col_mean_train = np.mean(input_matrix_train[:, col_idx])
        col_std_train = np.std(input_matrix_train[:, col_idx])
        # Directly act on self
        input_matrix_train[:, col_idx] = (input_matrix_train[:, col_idx] - col_mean_train) / col_std_train
        input_matrix_test[:, col_idx] = (input_matrix_test[:, col_idx] - col_mean_train) / col_std_train
    return input_matrix_train, input_matrix_test

