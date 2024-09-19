import os
import pandas as pd

gestures = ("empty","virtual_tap","virtual_slider_left","virtual_slider_right","swipe_left","still_hand")

train_df = pd.DataFrame(columns=["file_name","label"])
train_df["file_name"] = os.listdir("data/")
for idx, i in enumerate(os.listdir("data/")):
    
    for g in gestures:
        if g in i:
            train_df["label"][idx] = g

train_df.to_csv (r'gestures.csv', index = False, header=True)