import numpy as np
import os
from tqdm import tqdm
#Due to GPU compute limits , I have 2 dirs storing the .npy files of teh 10k generated images rather than one dir.
def save_to_npz2(input_dir1, input_dir2, output_file, max_images=10000):
    images = []  # Store all images

    # Load from input_dir1 first
    for file in tqdm(os.listdir(input_dir1), desc="Processing input_dir1"):
        if file.endswith(".npy"):
            img = np.load(os.path.join(input_dir1, file))  # Load image
            images.append(img)
        if len(images) == max_images:
            break  # Stop if we reach the limit

    # If not enough images, load from input_dir2
    if len(images) < max_images:
        for file in tqdm(os.listdir(input_dir2), desc="Processing input_dir2"):
            if file.endswith(".npy"):
                img = np.load(os.path.join(input_dir2, file))  # Load image
                images.append(img)
            if len(images) == max_images:
                break  # Stop if we reach the limit

    # Stack images and save as .npz
    if len(images) > 0:
        np.savez_compressed(output_file, images=np.stack(images))
        print(f"Saved dataset to {output_file} with {len(images)} images.")
    else:
        print("No images found!")