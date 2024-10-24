import os
import pandas as pd
import h5py

# gestures = ("virtual_tap","virtual_slider_left","virtual_slider_right","swipe_left","swipe_right", "swipe_up", "swipe_down",
#             "palm_grab", "palm_release", "virtual_knob_clockwise", "virtual_knob_anticlockwise", "pinch_out_horizontal", 
#             "pinch_out_vertical")
gestures = ("palm_release", "palm_grab", "swipe_up", "swipe_down", "swipe_right", "swipe_left")
# non_gestures = ("still_hand_open", "still_hand_closed", "empty")
# non_gestures = ("still_hand_open", )
non_gestures = ()

train_df = pd.DataFrame(columns=["file_name","label"])
test_new_df = pd.DataFrame(columns=["file_name","label"])
test_same_df = pd.DataFrame(columns=["file_name","label"])
all_df = pd.DataFrame(columns=["file_name","label"])


# train/val data:
root_dir = "data/"
for idx, filename in enumerate(os.listdir(root_dir)):
    
    file = h5py.File(os.path.join(root_dir, filename))
    num_frames = len(file['Sensors/TI_Radar/Data'].keys())

    if num_frames >= 20:
        for g in gestures:
            if g in filename:
                # Add to training df:
                loc = len(train_df.index)
                train_df.loc[loc, "file_name"] = filename
                train_df.loc[loc, "label"] = g
                # Add to all df:
                loc = len(all_df.index)
                all_df.loc[loc, "file_name"] = filename
                all_df.loc[loc, "label"] = g
        for n in non_gestures:
            if n in filename:
                # add to training df:
                loc = len(train_df.index)
                train_df.loc[loc, "file_name"] = filename
                train_df.loc[loc, "label"] = "non_gesture"
                # add to all df:
                loc = len(all_df.index)
                all_df.loc[loc, "file_name"] = filename
                all_df.loc[loc, "label"] = "non_gesture"

# test_new data:
root_dir = "data-test-new/"
for idx, filename in enumerate(os.listdir(root_dir)):
    
    file = h5py.File(os.path.join(root_dir, filename))
    num_frames = len(file['Sensors/TI_Radar/Data'].keys())

    if num_frames >= 20:
        for g in gestures:
            if g in filename:
                # Add to training df:
                loc = len(test_new_df.index)
                test_new_df.loc[loc, "file_name"] = filename
                test_new_df.loc[loc, "label"] = g
                # Add to all df:
                loc = len(all_df.index)
                all_df.loc[loc, "file_name"] = filename
                all_df.loc[loc, "label"] = g
        for n in non_gestures:
            if n in filename:
                # add to training df:
                loc = len(test_new_df.index)
                test_new_df.loc[loc, "file_name"] = filename
                test_new_df.loc[loc, "label"] = "non_gesture"
                # add to all df:
                loc = len(all_df.index)
                all_df.loc[loc, "file_name"] = filename
                all_df.loc[loc, "label"] = "non_gesture"

# test_same data:
root_dir = "data-test-same/"
for idx, filename in enumerate(os.listdir(root_dir)):
    
    file = h5py.File(os.path.join(root_dir, filename))
    num_frames = len(file['Sensors/TI_Radar/Data'].keys())

    if num_frames >= 20:
        for g in gestures:
            if g in filename:
                # Add to training df:
                loc = len(test_same_df.index)
                test_same_df.loc[loc, "file_name"] = filename
                test_same_df.loc[loc, "label"] = g
                # Add to all df:
                loc = len(all_df.index)
                all_df.loc[loc, "file_name"] = filename
                all_df.loc[loc, "label"] = g
        for n in non_gestures:
            if n in filename:
                # add to training df:
                loc = len(test_same_df.index)
                test_same_df.loc[loc, "file_name"] = filename
                test_same_df.loc[loc, "label"] = "non_gesture"
                # add to all df:
                loc = len(all_df.index)
                all_df.loc[loc, "file_name"] = filename
                all_df.loc[loc, "label"] = "non_gesture"


all_df.to_csv('csvs/all6.csv', index = False, header=True)
train_df.to_csv('csvs/train6.csv', index = False, header=True)
test_new_df.to_csv('csvs/test_new6.csv', index = False, header=True)
test_same_df.to_csv('csvs/test_same6.csv', index = False, header=True)


# gestures = ("virtual_tap","virtual_slider_left","virtual_slider_right","swipe_left","swipe_right", "swipe_up", "swipe_down",
#             "palm_grab", "palm_release", "virtual_knob_clockwise", "virtual_knob_anticlockwise", "pinch_out_horizontal", 
#             "pinch_out_vertical", "still_hand_open", "still_hand_closed", "empty")
# listed = []

# train_df = pd.DataFrame(columns=["file_name","label"])

# for g in gestures:
#     if g not in listed:
#         listed.append(g)
#         for idx, filename in enumerate(os.listdir("data/")):
#             if g in filename:
#                 train_df.loc[idx, "file_name"] = filename
#                 train_df.loc[idx, "label"] = g
#                 break

# train_df.to_csv ('smallset.csv', index = False, header=True)