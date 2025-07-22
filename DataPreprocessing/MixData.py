import os

def mix_real_and_fake(real_path, fake_path, output_path="ff.h5"):
    if not os.path.exists(real_path):
        raise FileNotFoundError(f"Real file not found: {real_path}")
    if not os.path.exists(fake_path):
        raise FileNotFoundError(f"Fake file not found: {fake_path}")

    with h5py.File(real_path, 'r') as f_real, h5py.File(fake_path, 'r') as f_fake:
        real_detections = f_real['detections'][:]
        fake_detections = f_fake['detections'][:]
        mixed_detections = np.concatenate([real_detections, fake_detections])
        with h5py.File(output_path, 'w') as f_out:
            f_out.create_dataset('detections', data=mixed_detections)

    print(f"✅ Mixed data saved to {output_path}")

mix_real_and_fake(
    real_path='DataPreprocessing/NormlizedData/sequence_2.h5',
    fake_path='Correct/Path/To/YourFile.h5',
    output_path='MixedData/sequence_2_mixed.h5'
)
