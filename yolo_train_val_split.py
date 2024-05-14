import os
import shutil

from sklearn.model_selection import train_test_split

TRAIN_SUB_DIR = "train"
VAL_SUB_DIR = "val"

VAL_SIZE = 0.1

RANDOM_SEED = 42

def move_files(file_paths, target_folder):
    if not os.path.exists(target_folder):
        os.makedirs(target_folder)
    for file_path in file_paths:
        filename = os.path.basename(file_path)
        shutil.move(file_path, os.path.join(target_folder, filename))

def main():
    images_dir = input("Enter the images directory path: ")
    labels_dir = input("Enter the labels directory path: ")

    labelled_image_paths = []
    unlabelled_image_paths = []
    label_paths = []

    print("Splitting images and labels into train and validation sets...")
    for image_filename in os.listdir(images_dir):
        image_name = image_filename.split('.')[0]
        label_file_path = os.path.join(labels_dir, f"{image_name}.txt")
        if os.path.exists(label_file_path):
            labelled_image_paths.append(os.path.join(images_dir, image_filename))
            label_paths.append(label_file_path)
        else:
            unlabelled_image_paths.append(os.path.join(images_dir, image_filename))

    train_image_paths, val_image_paths, train_label_paths, val_label_paths = train_test_split(labelled_image_paths, label_paths, test_size=VAL_SIZE, random_state=RANDOM_SEED)

    move_files(train_image_paths, os.path.join(images_dir, TRAIN_SUB_DIR))
    move_files(unlabelled_image_paths, os.path.join(images_dir, TRAIN_SUB_DIR))
    move_files(val_image_paths, os.path.join(images_dir, VAL_SUB_DIR))
    move_files(train_label_paths, os.path.join(labels_dir, TRAIN_SUB_DIR))
    move_files(val_label_paths, os.path.join(labels_dir, VAL_SUB_DIR))

    print("Images and labels splitted into train and validation sets.")

if __name__ == "__main__":
    main()
