import os
import numpy as np
import h5py
import matplotlib.pyplot as plt

class RadarScenesAnalyzer:
    def __init__(self, base_path, class_count=12):
        """
        Initializes the analyzer with the dataset path and number of classes.
        """
        self.base_path = base_path
        self.class_count = class_count
        self.class_counts = np.zeros(class_count, dtype=int)

    def process_sequences(self):
        """
        Processes all radar_data.h5 files and accumulates label_id counts.
        """
        for seq in sorted(os.listdir(self.base_path)):
            seq_path = os.path.join(self.base_path, seq)

            # Skip non-directory files like sensors.json
            if not os.path.isdir(seq_path):
                continue

            h5_path = os.path.join(seq_path, 'radar_data.h5')

            if not os.path.isfile(h5_path):
                print(f"Skipped {seq}: radar_data.h5 not found.")
                continue

            try:
                with h5py.File(h5_path, 'r') as h5f:
                    radar_data = h5f['radar_data']

                    if 'label_id' in radar_data.dtype.names:
                        label_ids = radar_data['label_id']
                        label_ids = label_ids[~np.isnan(label_ids)]
                        label_ids = label_ids.astype(int)
                        self.class_counts += np.bincount(label_ids, minlength=self.class_count)
                    else:
                        print(f"Skipped {seq}: 'label_id' field not found.")
            except Exception as e:
                print(f"Skipped {seq} due to error: {e}")

    def get_class_counts(self):
        """
        Returns the raw class count array.
        """
        return self.class_counts

    def plot_distribution(self):
        """
        Plots the class distribution as a percentage bar chart.
        """
        total = self.class_counts.sum()
        percentages = 100 * self.class_counts / total
        class_labels = [f"Class {i}" for i in range(self.class_count)]

        plt.figure(figsize=(12, 6))
        bars = plt.bar(class_labels, percentages, color='royalblue')
        plt.title("RadarScenes Class Distribution")
        plt.ylabel("Percentage (%)")
        plt.xticks(rotation=45)

        for bar, pct in zip(bars, percentages):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width() / 2, height + 0.5, f"{pct:.2f}%", ha='center')

        plt.tight_layout()
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.show()
    
    def get_missing_data(self):
        """
        Checks for missing (NaN) values across key radar fields for all sequences.
        Returns a dictionary showing the count of NaNs per field.
        """
        missing_counts = {}
        key_fields = ['x', 'y', 'z', 'rcs', 'label_id']  # Fields to check

        for field in key_fields:
            missing_counts[field] = 0  # initialize

        for seq in sorted(os.listdir(self.base_path)):
            seq_path = os.path.join(self.base_path, seq)

            if not os.path.isdir(seq_path):
                continue

            h5_path = os.path.join(seq_path, 'radar_data.h5')

            if not os.path.isfile(h5_path):
                continue

            try:
                with h5py.File(h5_path, 'r') as h5f:
                    radar_data = h5f['radar_data']
                    for field in key_fields:
                        if field in radar_data.dtype.names:
                            values = radar_data[field]
                            missing = np.isnan(values).sum()
                            missing_counts[field] += missing
            except Exception as e:
                print(f"Skipped {seq} due to error: {e}")

        print("\n🔍 Missing Data Summary:")
        for field, count in missing_counts.items():
            print(f"{field}: {count} NaNs")
        
        return missing_counts




# === Main entry ===
if __name__ == "__main__":
    base_path = 'NonStaticData' 

    analyzer = RadarScenesAnalyzer(base_path)
    analyzer.process_sequences()

    # counts = analyzer.get_class_counts()
    # print("Class counts:", counts)

    analyzer.plot_distribution()
    # analyzer.get_missing_data()
