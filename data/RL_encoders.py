
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
