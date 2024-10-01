import os
import pandas as pd
import h5py

gestures = ("virtual_tap","virtual_slider_left","virtual_slider_right","swipe_left","swipe_right", "swipe_up", "swipe_down",
            "palm_grab", "palm_release", "virtual_knob_clockwise", "virtual_knob_anticlockwise", "pinch_out_horizontal", 
            "pinch_out_vertical")
non_gestures = ("still_hand_open", "still_hand_closed", "empty")

train_df = pd.DataFrame(columns=["file_name","label"])
for idx, filename in enumerate(os.listdir("data/")):
    
    file = h5py.File(os.path.join('data', filename))

    num_frames = len(file['Sensors/TI_Radar/Data'].keys())

    if num_frames >= 20:
        loc = len(train_df.index)
        train_df.loc[loc, "file_name"] = filename



        for g in gestures:
            if g in filename:
                train_df.loc[loc, "label"] = g
        for n in non_gestures:
            if n in filename:
                train_df.loc[loc, "label"] = "non_gesture"

train_df.to_csv ('gestures_ge20.csv', index = False, header=True)

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