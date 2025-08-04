import h5py
import sys
import os

def inspect_h5_file(file_path):
    if not os.path.exists(file_path):
        print(f"❌ File not found: {file_path}")
        return

    try:
        with h5py.File(file_path, 'r') as f:
            print(f"\n📂 Inspecting: {file_path}")
            print("Available keys and shapes:")
            for key in f.keys():
                data = f[key]
                print(f" - {key}: shape {data.shape}, dtype {data.dtype}")
    except Exception as e:
        print(f"❌ Failed to read H5 file: {e}")

if __name__ == "__main__":
    # 👇 Replace this with your own file path
    file_path = "FusedData/sequence_1_fused.h5"
    inspect_h5_file(file_path)
