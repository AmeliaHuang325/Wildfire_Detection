# Wildfire Detection Project

## Overview
This project focuses on wildfire detection using deep learning models. It includes data preprocessing, model training, and testing, structured into four main Jupyter Notebooks.

## Project Structure
```
├── archive                   # Folder containing backup or previous versions of files
├── 00_Image labeling.ipynb              # Draw annotation bounding boxs and save as YOLO format
├── 01_Convert XML to YOLO Format.ipynb  # Converts annotations from XML to YOLO format
├── 02_Model training.ipynb              # Trains a YOLO-based wildfire detection model
├── 03_Test.ipynb                         # Tests and evaluates the trained model
├── datasets\Whildfire photos            # Dataset folder containing images and labels
│   ├── images
│   │   ├── train                        # 80% of images
│   │   ├── val                          # 10% of images
│   │   ├── test                         # 10% of images
│   ├── labels
│   │   ├── train                        # YOLO labels for train images
│   │   ├── val                          # YOLO labels for val images
│   │   ├── test                         # YOLO labels for test images
│   ├── wildfire.yaml                    # YOLO dataset configuration file
├── Wildfire photos                       # Photos downloaded from Kaggle
│   ├── Annotations                       # Kaggle annotations in XML format
│   ├── Datacluster Fire and Smoke Sample # Kaggle images
│   ├── YOLO labels                       # Modified labels from XML to YOLO format
├── wildfire_aerial.jpg                   # Sample image for annotation
├── wildfire_aerial.txt                   # Sample annotation file
├── forest_fire.jpg                       # Test file
├── output_0.jpg                          # Test file - output
└── README.md                             # Project documentation
```
The wildfire photos are downloaded from: https://www.kaggle.com/datasets/dataclusterlabs/fire-and-smoke-dataset

## Requirements
Ensure you have the following dependencies installed:

```bash
pip install ultralytics opencv-python numpy pandas torch torchvision
```

## Notebook Details

### 0. Image Labeling
- Annotates wildfire images for training the detection model.
- Labels images using bounding boxes to identify fire-affected areas.

### 1. Convert XML to YOLO Format
- Converts dataset annotations from **Pascal VOC XML format** to **YOLO format**.
- Saves the converted labels in the required YOLO structure.

### 2. Model Training
- Loads the YOLO-formatted dataset.
- Configures and trains a **YOLO model** for wildfire detection.
- Utilizes GPU acceleration if available.

### 3. Model Testing
- Loads the trained model.
- Runs inference on test images.
- Evaluates model performance using metrics like precision and recall.

## Running the Project
1. **Prepare the Dataset**: Ensure all images and labels are in the correct YOLO format.
2. **Train the Model**: Open and run `02_Model training.ipynb`.
3. **Test the Model**: Open `03_Test.ipynb` to evaluate the trained model on test images.

## Notes
- Ensure the dataset path is correctly configured in the training and testing notebooks.

## Contributions
Feel free to fork this repository and submit pull requests for improvements!

## License
This project is open-source under the [MIT License](LICENSE).