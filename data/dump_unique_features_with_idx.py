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

# Todo: some of the features needs tokenizing maybe >> BUT if you do this you also need to change  one_cold_feature_encoder()
# Some of them are given like:
# GoTo Hell Example: "Gym Fitness Lounge Swimming Pool Sun Decks Exercise Studios Indoor Half-Basketball Court" --> Used Only Once
# GoTo Hell Example: "** HOLY DEAL BATMAN! * ENTIRE FLOOR! * MASSIVE 4BR MANSION * GOURMET KITCHEN * PETS OK **" --> Used Only Once
# Reasonable Example: "private-outdoor-space" --> Used 22 Times
# Typo Example: "Diswasher"  --> Used Only Once :)
# Good Example: "Doorman"  --> Used 20898 Times
