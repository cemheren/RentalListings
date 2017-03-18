import ijson
import json
import numpy as np


def find_keep_mask_for_price(price_npv, upper_perc=None, lower_perc=None):
    # price_npv is a numpy array (single column)
    if upper_perc is None:
        upper_perc = 99.95
    if lower_perc is None:
        lower_perc = 0.005
    # Eval Percentile Limit Values
    llimit = np.percentile(price_npv, lower_perc)
    ulimit = np.percentile(price_npv, upper_perc)
    price_keep_mask = (price_npv < ulimit) & (price_npv > llimit)
    # This price_keep_mask is a boolean list for the entries to be kept
    return price_keep_mask


def normalize_cols(input_matrix_train, input_matrix_test, normalizableColumnResolver, col_name_list):
    for col_name in col_name_list:
        col_idx = normalizableColumnResolver[col_name]
        # Trim A bit
        len_data = len(input_matrix_train[:, col_idx])
        sorted_train_data = np.array(input_matrix_train[:, col_idx])
        sorted_train_data = np.sort(sorted_train_data)
        llimit = int(0.05 * len_data)
        ulimit = int(0.95 * len_data)

        temp_sorted_trimmed = sorted_train_data[llimit:ulimit]

        # Get mean median hybrid shift
        col_mean_train = np.mean(temp_sorted_trimmed)
        col_median_train = np.median(temp_sorted_trimmed)
        col_shift = (col_mean_train + col_median_train) / 2

        # Get Stdev
        #col_std_train = np.std(input_matrix_train[:, col_idx])
        col_std_train = np.std(temp_sorted_trimmed)

        # Directly act on self (since it is pass-by-reference)
        input_matrix_train[:, col_idx] = (input_matrix_train[:, col_idx] - col_shift) / col_std_train
        input_matrix_test[:, col_idx] = (input_matrix_test[:, col_idx] - col_shift) / col_std_train
    return input_matrix_train, input_matrix_test

