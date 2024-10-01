import h5py
import os
import pandas as pd

info_df = pd.DataFrame(columns=["file_name", "num_frames"])

for idx, filename in enumerate(os.listdir("data/")):

    file = h5py.File(os.path.join('data', filename))

    info_df.loc[idx, "file_name"] = filename

    num_frames = len(file['Sensors/TI_Radar/Data'].keys())

    info_df.loc[idx, "num_frames"] = num_frames

    if num_frames <25:
        print(filename[35:-1], ":", num_frames)

info_df.to_csv (r'gestures_info.csv', index = False, header=True)  
