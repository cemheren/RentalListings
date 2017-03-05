import operator
import ijson
import json
from collections import Counter

f = open('train.json', 'r')
f2write_idx = open('ufeature_to_index_dictionary.json', 'w')
f2write_cnt = open('ufeature_to_counter_dictionary.json', 'w')

json_decode = json.load(f)
features = json_decode['features']


feature_list = []
unique_feature_ap_count = Counter()
for item in features:
    this_items_features = features[item]
    unique_feature_ap_count.update(this_items_features)
    for single_feature in this_items_features:
        feature_list.append(single_feature)

set_of_features = set(feature_list)
unique_feature_list = list(set_of_features)
unique_feature_idx_dict = {unique_feature_list[ii]: ii for ii in range(0, len(unique_feature_list))}

print unique_feature_ap_count.most_common(7)

json.dump(unique_feature_idx_dict, f2write_idx)
f2write_idx.close()

json.dump(unique_feature_ap_count, f2write_cnt)
f2write_cnt.close()
