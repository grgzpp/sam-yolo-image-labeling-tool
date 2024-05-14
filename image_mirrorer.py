import os

import cv2
from tqdm import tqdm

def flip_labels(label_file_path):
    with open(label_file_path, 'r') as file:
        lines = file.readlines()

    flipped_labels = []
    for line in lines:
        data = line.strip().split()
        if len(data) > 0:
            label_id = int(data[0])
            x_center = float(data[1])
            y_center = float(data[2])
            width = float(data[3])
            height = float(data[4])

            x_center_flipped = 1.0 - x_center
            y_center_flipped = y_center
            flipped_labels.append(f"{label_id} {x_center_flipped:.6f} {y_center_flipped:.6f} {width:.6f} {height:.6f}")
    
    return flipped_labels

def main():
    images_dir = input("Enter the images directory path: ")
    labels_dir = input("Enter the labels directory path: ")

    print("Mirroring images and labels...")
    for image_filename in tqdm(os.listdir(images_dir)):
        image_name = image_filename.split('.')[0]
        label_file_path = os.path.join(labels_dir, f"{image_name}.txt")
        if os.path.exists(label_file_path):
            image = cv2.imread(os.path.join(images_dir, image_filename))
            cv2.imwrite(os.path.join(images_dir, f"{image_name}_mirrored.jpg"), cv2.flip(image, 1))

            flipped_labels = flip_labels(label_file_path)
            with open(os.path.join(labels_dir, f"{image_name}_mirrored.txt"), 'w') as f:
                for label in flipped_labels:
                    f.write(label + '\n')

    print("Data augmentation (mirroring) complete.")

if __name__ == "__main__":
    main()
