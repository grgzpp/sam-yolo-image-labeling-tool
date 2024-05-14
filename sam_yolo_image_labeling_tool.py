import os

import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.patches as patches
import torch
from tqdm import tqdm
from segment_anything import SamPredictor, sam_model_registry

CHECHPOINT_PATH = os.path.join("weights", "sam_vit_l_0b3195.pth")
MODEL_TYPE = "vit_l"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
np.random.seed(42)

class Selector:

    def __init__(self, images_dir, labels_dir):
        self.images_dir = images_dir
        self.labels_dir = labels_dir

        try:
            with open(os.path.join(self.images_dir, "..", "labels.txt"), 'r') as f:
                self.defined_labels = f.readlines()
        except:
            raise Exception("File labels.txt missing in the parent directory of input directory")
        
        if not os.path.exists(labels_dir):
            os.makedirs(labels_dir)
            
        self.label_colors = [np.random.random(3) for _ in range(len(self.defined_labels))]
        self.handles = [mlines.Line2D([], [], color=self.label_colors[i], label=f"{i}: {self.defined_labels[i]}") for i in range(len(self.defined_labels))]
        
        total_image_filenames = os.listdir(images_dir)
        self.to_label_image_filenames = []
        for image_filename in total_image_filenames:
            if not os.path.exists(os.path.join(self.labels_dir, f"{image_filename.split('.')[0]}.txt")):
                self.to_label_image_filenames.append(image_filename)
        print(f"Total images found in specified directory: {len(total_image_filenames)}. Images to be labelled: {len(self.to_label_image_filenames)}")

        self.total_bboxes = []
        self.total_label_ids = []
        for _ in range(len(self.to_label_image_filenames)):
            self.total_label_ids.append([])
            self.total_bboxes.append([])

        sam = sam_model_registry[MODEL_TYPE](checkpoint=CHECHPOINT_PATH).to(device)
        self.predictor = SamPredictor(sam)
        print("SAM predictor initialized")

        self.cursor = 0
        self.current_bboxes = []
        self.current_label_ids = []
        self.composed_number = ""

        self.fig, self.ax = plt.subplots()
        self.rect = None
        self.start_point = None
        self.end_point = None

        self.ax.axis("on")
        self.fig.canvas.mpl_connect("button_press_event", self.on_click)
        self.fig.canvas.mpl_connect("key_press_event", self.on_key_press)
        self.fig.canvas.mpl_connect("motion_notify_event", self.on_motion)
        self.fig.canvas.mpl_connect("button_release_event", self.on_release)
        self.load_current_image()
        plt.show()

    def load_current_image(self):
        self.current_image_filename = self.to_label_image_filenames[self.cursor]
        self.current_image = cv2.imread(os.path.join(self.images_dir, self.current_image_filename))
        self.current_image = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2RGB)
        self.predictor.set_image(self.current_image)
        if self.total_bboxes[self.cursor] and self.total_label_ids[self.cursor]:
            self.current_bboxes = list(self.total_bboxes[self.cursor])
            self.current_label_ids = list(self.total_label_ids[self.cursor])
        self.draw_current_labels()
        print(f"Loaded image ({self.cursor + 1}/{len(self.to_label_image_filenames)}): {self.current_image_filename}")

    def save_bboxes_to_yolo_format(self, image_width, image_height, bboxes, label_ids, output_file):
        if len(bboxes) != len(label_ids):
            print("Length of bboxes and labels must be equal")
            return

        with open(output_file, 'w') as f:
            for i in range(len(bboxes)):
                x_min, y_min, x_max, y_max = bboxes[i]
                x_center = (x_min + x_max)/2
                y_center = (y_min + y_max)/2
                x_center_norm = x_center/image_width
                y_center_norm = y_center/image_height
                width_norm = (x_max - x_min)/image_width
                height_norm = (y_max - y_min)/image_height

                yolo_format_string = f"{label_ids[i]} {x_center_norm:.6f} {y_center_norm:.6f} {width_norm:.6f} {height_norm:.6f}"
                f.write(yolo_format_string + '\n')

    def draw_mask(self, mask):
        color = np.array([30/255, 144/255, 255/255, 0.6])
        h, w = mask.shape[-2:]
        mask_image = mask.reshape(h, w, 1)*color.reshape(1, 1, -1)
        self.ax.imshow(mask_image)

    def draw_bbox(self, bbox, label_id, color):
        x0, y0 = bbox[0], bbox[1]
        w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
        self.ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor=color, facecolor=(0, 0, 0, 0), lw=2))
        self.ax.text(x0, y0 - 2, label_id, color=color, fontsize=10, ha="left", va="bottom")

    def draw_current_labels(self):
        self.ax.cla()
        self.rect = None
        self.ax.set_title(self.current_image_filename)
        self.ax.legend(handles=self.handles, loc=1, bbox_to_anchor=(1.25, 1.0))
        self.ax.imshow(self.current_image)
        for i in range(len(self.current_bboxes)):
            self.draw_bbox(self.current_bboxes[i], self.current_label_ids[i], self.label_colors[self.current_label_ids[i]])
        plt.draw()

    def find_mask_bbox(self, mask):
        rows, cols = np.nonzero(mask)
        x_min, y_min = np.min(cols), np.min(rows)
        x_max, y_max = np.max(cols), np.max(rows)
        return (x_min, y_min, x_max, y_max)

    def register_current_labels(self):
        self.total_bboxes[self.cursor] = list(self.current_bboxes)
        self.total_label_ids[self.cursor] =list(self.current_label_ids)
        print(f"Registered labels for image {self.to_label_image_filenames[self.cursor]}: {self.total_label_ids[self.cursor]}")

    def reset_current_labels(self):
        self.current_bboxes.clear()
        self.current_label_ids.clear()
        self.composed_number = ""

    def on_click(self, event):
        if event.button == 1 and event.xdata and event.ydata:
            self.start_point = (event.xdata, event.ydata)
            self.composed_number = ""

    def on_motion(self, event):
        if self.start_point is not None and event.xdata and event.ydata:
            if self.rect:
                self.rect.remove()
            self.end_point = (event.xdata, event.ydata)
            self.rect = patches.Rectangle(self.start_point, self.end_point[0] - self.start_point[0], self.end_point[1] - self.start_point[1], linewidth=2, edgecolor='b', facecolor="none")
            self.ax.add_patch(self.rect)
            plt.draw()

    def on_release(self, event):
        if event.button == 1 and event.xdata and event.ydata:
            if self.rect:
                self.rect.remove()
            self.end_point = (event.xdata, event.ydata)
            if self.start_point != self.end_point:
                x_min, x_max = sorted([self.start_point[0], self.end_point[0]])
                y_min, y_max = sorted([self.start_point[1], self.end_point[1]])
                drawed_box = np.array([x_min, y_min, x_max, y_max])
                masks, _, _ = self.predictor.predict(
                    point_coords=None,
                    point_labels=None,
                    box=drawed_box[None, :],
                    multimask_output=False,
                )
                mask = masks[0]
                bbox = self.find_mask_bbox(mask)
                self.current_bboxes.append(bbox)
                self.current_label_ids.append(0)
                self.draw_current_labels()
                self.draw_mask(mask)
                
            self.start_point = None
            self.end_point = None
            self.rect = None

    def on_key_press(self, event):
        pressed_key = event.key.lower()
        if pressed_key == "right":
            self.register_current_labels()
            self.reset_current_labels()
            if self.cursor < (len(self.to_label_image_filenames) - 1):
                self.cursor += 1
                self.load_current_image()
            else:
                print("Last image labelled, press 'c' to save all labels in YOLO format")
        elif pressed_key == "left":
            self.register_current_labels()
            self.reset_current_labels()
            if self.cursor > 0:
                self.cursor -= 1
                self.load_current_image()
        elif pressed_key == 'd':
            if len(self.current_bboxes) > 0:
                self.current_bboxes.pop()
                removed_label = self.current_label_ids.pop()
                print(f"Deleted last label: {removed_label}")
                self.draw_current_labels()
        elif pressed_key in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']:
            self.composed_number += pressed_key
            number = int(self.composed_number)
            if number < len(self.defined_labels):
                self.current_label_ids[-1] = number
            else:
                print(f"Composed number out of range (last label is {len(self.defined_labels) - 1}), set back to 0")
                self.current_label_ids[-1] = 0
                self.composed_number = ""
            self.draw_current_labels()
        elif pressed_key == 'c':
            print("Saving all labels in YOLO format...")
            if not os.path.exists(self.labels_dir):
                os.makedirs(self.labels_dir)
            saved_labels = 0
            for i in tqdm(range(len(self.total_bboxes))):
                bboxes = self.total_bboxes[i]
                if not bboxes:
                    continue
                label_ids = self.total_label_ids[i]
                image_filename = self.to_label_image_filenames[i]
                image = cv2.imread(os.path.join(self.images_dir, image_filename))
                h, w = image.shape[0], image.shape[1]
                output_file = os.path.join(self.labels_dir, f"{image_filename.split('.')[0]}.txt")
                self.save_bboxes_to_yolo_format(w, h, bboxes, label_ids, output_file)
                saved_labels += 1

            print(f"Saved labels for {saved_labels}/{len(self.total_bboxes)} images")
            plt.close(self.fig)

def main():
    images_dir = input("Enter the images directory path: ")
    labels_dir = input("Enter the labels directory path: ")

    Selector(images_dir, labels_dir)

if __name__ == "__main__":
    main()
