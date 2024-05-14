import os

import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.patches as patches

def get_bbox_from_yolo_format(image, yolo_format_string):
    image_height, image_width = image.shape[0], image.shape[1]

    data = yolo_format_string.split()
    
    label_id = int(data[0])
    
    x_center_norm, y_center_norm, width_norm, height_norm = map(float, data[1:])
    x_center = x_center_norm*image_width
    y_center = y_center_norm*image_height
    width = width_norm*image_width
    height = height_norm*image_height
    x_min = int(x_center - (width/2))
    y_min = int(y_center - (height/2))
    x_max = int(x_center + (width/2))
    y_max = int(y_center + (height/2))
    
    bbox = (x_min, y_min, x_max, y_max)
    
    return bbox, label_id

def on_key_press(event):
    if event.key.lower() == 'b':
        exit()

def main():
    images_dir = input("Enter the images directory path: ")
    labels_dir = input("Enter the labels directory path: ")

    try:
        with open(os.path.join(images_dir, "..", "labels.txt"), 'r') as f:
            defined_labels = f.readlines()
    except:
        raise Exception("File labels.txt missing in the parent directory of input directory")
        
    label_colors = [np.random.random(3) for _ in range(len(defined_labels))]
    handles = [mlines.Line2D([], [], color=label_colors[i], label=defined_labels[i]) for i in range(len(defined_labels))]

    for image_filename in os.listdir(images_dir):
        image_name = image_filename.split('.')[0]
        image = cv2.imread(os.path.join(images_dir, f"{image_name}.jpg"))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        fig, ax = plt.subplots()
        fig.canvas.mpl_connect("key_press_event", on_key_press)
        ax.set_title(image_filename)
        ax.legend(handles=handles, loc=1)
        ax.imshow(image)

        with open(os.path.join(labels_dir, f"{image_name}.txt"), 'r') as f:
            labels = f.readlines()
            for yolo_format_string in labels:
                bbox, label_id = get_bbox_from_yolo_format(image, yolo_format_string)
                x_min, y_min, x_max, y_max = bbox
                bbox_width = x_max - x_min
                bbox_height = y_max - y_min
                rect = patches.Rectangle((x_min, y_min), bbox_width, bbox_height, linewidth=2, edgecolor=label_colors[label_id], facecolor="none")
                ax.add_patch(rect)
                ax.text(x_min, y_min - 2, label_id, color=label_colors[label_id], fontsize=10, ha='left', va='bottom')
                
        plt.show()

if __name__ == "__main__":
    main()
    