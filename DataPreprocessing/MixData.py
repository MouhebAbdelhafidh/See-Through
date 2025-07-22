import os
import h5py
import numpy as np

def mix_all_real_and_fake(real_dir, fake_base_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    
    # Get list of real sequence files
    real_files = [f for f in os.listdir(real_dir) if f.endswith('.h5')]

    # Get list of all fake label folders (e.g., Label1, Label2, etc.)
    fake_label_dirs = [
        os.path.join(fake_base_dir, d)
        for d in os.listdir(fake_base_dir)
        if os.path.isdir(os.path.join(fake_base_dir, d))
    ]

    for real_file in real_files:
        sequence_name = real_file.replace('.h5', '')  # e.g., "sequence_1"
        real_path = os.path.join(real_dir, real_file)
        
        # Load real detections
        with h5py.File(real_path, 'r') as f_real:
            real_detections = f_real['detections'][:]

        # Collect all corresponding fake files for this sequence
        fake_detections_list = []
        for label_dir in fake_label_dirs:
            fake_file = f"{sequence_name}_fake_{os.path.basename(label_dir).lower()}.h5"
            fake_path = os.path.join(label_dir, fake_file)
            if os.path.exists(fake_path):
                with h5py.File(fake_path, 'r') as f_fake:
                    fake_detections_list.append(f_fake['detections'][:])
            else:
                print(f"⚠️ Missing fake file: {fake_path}")

        # Concatenate all real and fake detections
        all_detections = np.concatenate([real_detections] + fake_detections_list)

        # Write to output file
        output_path = os.path.join(output_dir, f"{sequence_name}_mixed.h5")
        with h5py.File(output_path, 'w') as f_out:
            f_out.create_dataset('detections', data=all_detections)

        print(f"✅ Mixed data saved to {output_path}")

# Usage:
mix_all_real_and_fake(
    real_dir='../NormlizedData',
    fake_base_dir='FakeData',
    output_dir='../MixedData'
)
