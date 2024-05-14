# SAM-YOLO Image Labeling Tool

This Python tool provides an easy way to label objects in images with bounding boxes for YOLO training. It allows you to load images from a directory, draw bounding boxes around objects of interest, assign labels, and save the labels in YOLO format. It is powered by the amazing Segment Anything Model (SAM), by Meta AI. Follow the instructions on their [GitHub repo](https://github.com/facebookresearch/segment-anything) for the installation. It allows to get precise bounding boxes around objects without much effort in drawing them, as this model segments the most likely element inside the drawn bounding box.

Make sure to donwload a SAM model weights (from the GitHub repo) and specify it in *MODEL_TYPE* parameter. Also make sure to specify the right path to where you savde the weights file *CHECHPOINT_PATH*. The labeling tool has been tested with ViT-L SAM model (large model) and works really weel. It is a little bit slow, depending on the hardware, but the SAM model is really heavy, completely understandable for what it can do.

This tool only produces bounding boxes and not masks, but SAM actually extracts masks from the items, so it would be really easy to extract and save them. I will update this feature if I need it in the future or if you ask for it. You can also visualize produced labels using the provided labels visualizer, to ensure everything is fine.

The tool also comes with an image mirrorer to be executed AFTER the labelling, for data augmentation. It also mirrors also the corresponding labels. The usage is really intuitive and the prerequisites are the same of the main tool.

Once you have labelled and eventually mirrored the data, you can run the train-validation splitter. It works on the same folders you created for the images and labels, just specify them in the program prompt. You can set the validation size in % and the random seed.

## Prerequisites

- Python 3.x
- PyTorch (CUDA installation is highly recommended, follow the steps on PyTorch [documentation](https://pytorch.org/))
- Segment Anything Model (SAM)
- OpenCV
- Matplotlib
- tqdm

## Installation

Clone this repository and install the required dependencies with:
```bash
pip install -r requirements.txt
```
## Usage

1. Edit the *labels.txt* file with the different objects you are labelling, one per line. The indices will be automatically applied following the order in this list This file must be placed in the same folder of the Python script.

2. Run the script:
    ```bash
    python sam_yolo_image_labeling_tool.py
    ```

3. Enter the path to the directory containing the images when prompted and the path to the directory where you want to save the labels.

4. Label the objects in the images:
   - Click and drag to draw bounding boxes around objects.
   - Press keys 0 to 9 to assign labels to the boxes (you can also compose number >9; if the last specified label index is exceeded, it will be reset to 0).
   - Press d to delete the last label.
   - Press left arrow to move to the previous image.
   - Press right arrow to move to the next image.
   - Press c to save the labels in YOLO format in the specified labels folder.
   - Press q to quit without saving.

5. [Optional] Run the *image_mirrorer.py* script to mirror the images and the associated labels you just created. This is for data augmentation.

6. [Optional] Run the *yolo_train_val_split.py* script to execute the train-validation split, needed to train YOLO. Make sure to set the validation size in % and the random seed as desired.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
