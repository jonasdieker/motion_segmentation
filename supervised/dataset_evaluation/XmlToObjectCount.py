import os
import json
import xmltodict

def get_per_frame_count(xml_root):
    with open(os.path.join(xml_root, "tracklet_labels.xml"), "r") as f:
        xml_data = f.read()

    dict_data = xmltodict.parse(xml_data)
    str_data = json.dumps(dict_data)
    json_data = json.loads(str_data)
    print(f'Number of tracklets: {json_data["boost_serialization"]["tracklets"]["count"]}')

    # get list of tracklets
    tracklets = json_data["boost_serialization"]["tracklets"]["item"]
    # print(tracklets)

    # build dict
    object_count = {}
    for i in range(len(tracklets)):
        # print(tracklets[i]["objectType"])
        first_frame = int(tracklets[i]["first_frame"])
        final_frame = first_frame + int(tracklets[i]["poses"]["count"])
        for j in range(first_frame, final_frame):
            if j not in object_count.keys():
                object_count[j] = 1
            if j in object_count.keys():
                object_count[j] += 1

    return object_count


# specify desired sequences
dirs_nums = ["0005", "0013", "0014", "0015", "0018", "0032", "0051", "0056", "0057", "0059", "0060", "0084"]
tracklet_root = "/storage/remote/atcremers40/motion_seg/datasets/Extended_MOD_Masks/tracklets/2011_09_26/"

# loop over all desired sequence nums
dataset_object_count = {}
for sequence_num in dirs_nums:
    xml_root = os.path.join(tracklet_root, f"2011_09_26_drive_{sequence_num}_sync")
    object_count = get_per_frame_count(xml_root)
    dataset_object_count[sequence_num] = object_count
    print(f"added sequence {sequence_num} counts")

# specify where to save dict
save_path = "/storage/remote/atcremers40/motion_seg/datasets/Extended_MOD_Masks/tracklets"
with open(os.path.join(save_path, "object_count_per_frame.json"), "w") as f:
    json.dump(dataset_object_count, f)

print("\nwritten json to remote!")