import os
import subprocess

# Configuration
input_folder = "MixedData"
output_folder = "FusedData"
radius = 0.5  
weights = "0:1.0" 
fusion_script = "early_fusion.py"

# List all input HDF5 files
all_files = sorted([
    f for f in os.listdir(input_folder)
    if f.endswith("_mixed.h5")
])

print(f"Found {len(all_files)} sequence files to process.\n")

for i, filename in enumerate(all_files):
    input_path = os.path.join(input_folder, filename)
    base_name = filename.replace("_mixed.h5", "")
    output_filename = f"{base_name}_fused.h5"
    output_path = os.path.join(output_folder, output_filename)

    # Skip already processed files
    if os.path.exists(output_path):
        print(f"[{i+1}/{len(all_files)}] Skipping {filename} (already fused).")
        continue

    # Build the command
    cmd = [
        "python", fusion_script,
        "--in", input_path,
        "--out", output_path,
        "--radius", str(radius)
    ]

    if weights:
        cmd += ["--weights", weights]

    print(f"[{i+1}/{len(all_files)}] Processing {filename}...")
    try:
        subprocess.run(cmd, check=True)
        print(f"✅ Finished: {output_filename}\n")
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to fuse {filename}: {e}\n")
        break
