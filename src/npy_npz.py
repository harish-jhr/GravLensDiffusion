import numpy as np
import os
from tqdm import tqdm
'''
    This module is to save all .npy image sample files into a single .npz file, this is 
    faster to load images(with past experience)
'''
def save_to_npz(input_dir, output_file):
    data_dict = {}
    
    for cls in os.listdir(input_dir):  # Loop through class folders
        class_path = os.path.join(input_dir, cls)
        if os.path.isdir(class_path):
            images = []
            for file in tqdm(os.listdir(class_path), desc=f"Processing {cls}"):
                if file.endswith(".npy"):
                    img = np.load(os.path.join(class_path, file))  # Load image
                    images.append(img)  # Append to list
            data_dict[cls] = np.stack(images)  # Stack images for this class

    np.savez_compressed(output_file, **data_dict)  # Save as .npz
    print(f"Saved dataset to {output_file}")

