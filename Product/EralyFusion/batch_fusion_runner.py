import os
import subprocess
import json

# Paths
input_folder = "MixedData"
output_folder = "FusedData"
log_file = "fused_log.txt"
error_log_file = "failed_log.json"

# Fusion parameters
radius = 0.5
weights = "0:1.0"  # Example, adjust as needed

# Load already fused files
fused_set = set()
if os.path.exists(log_file):
    with open(log_file, "r") as f:
        fused_set = set(line.strip() for line in f)

# Load previously failed log
failed_log = {}
if os.path.exists(error_log_file):
    with open(error_log_file, "r") as f:
        failed_log = json.load(f)

# Go through each .h5 file in input folder
all_files = sorted(os.listdir(input_folder))
for fname in all_files:
    if not fname.endswith(".h5") or fname in fused_set:
        continue

    input_path = os.path.join(input_folder, fname)
    output_fname = fname.replace("_mixed.h5", "_fused.h5")
    output_path = os.path.join(output_folder, output_fname)

    cmd = [
        "python", "early_fusion.py",
        "--in", input_path,
        "--out", output_path,
        "--radius", str(radius),
        "--weights", weights
    ]

    print(f"▶️ Processing {fname}...")
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(f"✅ Success: {fname}")

        # Log success
        with open(log_file, "a") as f:
            f.write(fname + "\n")

    except subprocess.CalledProcessError as e:
        error_message = e.stderr.strip().split("\n")[-1]
        print(f"❌ Failed: {fname} — {error_message}")

        # Log failure with reason
        failed_log[fname] = error_message

        # Update error log
        with open(error_log_file, "w") as f:
            json.dump(failed_log, f, indent=2)
