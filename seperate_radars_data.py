import h5py
import numpy as np
import os
import glob

H5_FOLDER = "MixedData"
OUTPUT_DIR = "SplitBySensor"

os.makedirs(OUTPUT_DIR, exist_ok=True)

sensor_data_dict = {}

for h5_file in glob.glob(os.path.join(H5_FOLDER, "*.h5")):
    print(f"Processing file: {h5_file}")
    with h5py.File(h5_file, "r") as f:
        data = f["detections"][()]  
        
        sensor_ids = np.unique(data["sensor_id"])
        
        for sid in sensor_ids:
            sid_data = data[data["sensor_id"] == sid]
            
            if sid in sensor_data_dict:
                sensor_data_dict[sid] = np.concatenate((sensor_data_dict[sid], sid_data))
            else:
                sensor_data_dict[sid] = sid_data

# Save all sensor data 
for sid, sid_data in sensor_data_dict.items():
    sensor_folder = os.path.join(OUTPUT_DIR, f"sensor_{sid}")
    os.makedirs(sensor_folder, exist_ok=True)
    
    output_file = os.path.join(sensor_folder, f"sensor_{sid}_detections.h5")
    with h5py.File(output_file, "w") as out_f:
        out_f.create_dataset("detections", data=sid_data)
    
    print(f"Saved {len(sid_data)} records for sensor {sid} to {output_file}")
