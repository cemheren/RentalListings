import ijson
import json
import numpy as np
import re
import string
from collections import Counter

# Arguments:
# features_data_dict ==>  Dictionary that contains {listing_id: feature_list} pairs  [from json file]
# unique_feature_idx_dict ==> Dictionary that contains {feature_text: feature_fixed_index} pairs [from json file dump]
def to_one_cold_feature_encoding(features_data_dict, unique_feature_idx_dict):
    num_unique_features = len(unique_feature_idx_dict)
    encoded_features_dict = dict()
    for item in features_data_dict:
        this_items_features = features_data_dict[item]
        indices_to_set_1_for_this_item = []
        for single_feature in this_items_features:
            indices_to_set_1_for_this_item.append(unique_feature_idx_dict[single_feature])
        encoded_features = [0] * num_unique_features
        for idx_to_set_1 in indices_to_set_1_for_this_item:
            encoded_features[idx_to_set_1] = 1
        # Now set this encoding to listing 'item'
        encoded_features_dict[item] = encoded_features
    return encoded_features_dict


def to_region_idx_encoding(loaded_x_json):
    _roi_lat_min = 40.44
    _roi_lat_max = 40.94
    _roi_long_min = -74
    _roi_long_max = -73
    _roi_xisting_idxs = [0, 21, 22, 23, 31, 32, 33, 41, 42, 43, 51, 52, 53, 61, 62, 63, 71, 72, 81, 82, 91, 92, 93, 98]
    _roi_xisting_idxs_dict = {_roi_xisting_idxs[ii]: ii for ii in range(0, len(_roi_xisting_idxs))}

    latitude = [v for v in loaded_x_json['latitude'].values()]
    longitude = [v for v in loaded_x_json['longitude'].values()]

    num_entries = len(latitude)

    # Just convert and put region Id // ZERO means outside
    region_xids = [0] * num_entries
    for ii in range(0, num_entries):
        curr_lat = latitude[ii]
        curr_long = longitude[ii]
        if (_roi_lat_min < curr_lat and curr_lat < _roi_lat_max) and (_roi_long_min < curr_long and curr_long < _roi_long_max):
            discrete_height = np.floor((curr_lat - _roi_lat_min) / 0.05)
            discrete_widthh = np.floor((curr_long - _roi_long_min) / 0.1)
            raw_region_index = discrete_height * 10 + discrete_widthh + 1
            # In Region of Interest ==> Assign a sub region index from dictionary
            if _roi_xisting_idxs_dict.has_key(raw_region_index):
                region_xids[ii] = _roi_xisting_idxs_dict[raw_region_index]
    return region_xids


def get_top_n_features(x_json, number_of_features):
    features = x_json['features']
    feature_list = []
    unique_feature_ap_count = Counter()
    for item in features:
        this_items_features = [re.sub(r'[^\w]', '', v.translate(string.punctuation).lower()) for v in features[item]]
        unique_feature_ap_count.update(this_items_features)
        for single_feature in this_items_features:
            feature_list.append(single_feature)

    set_of_features = set(feature_list)
    unique_feature_list = list(set_of_features)
    unique_feature_idx_dict = {unique_feature_list[ii]: ii for ii in range(0, len(unique_feature_list))}

    most_common_features = unique_feature_ap_count.most_common(number_of_features)

    return most_common_features
