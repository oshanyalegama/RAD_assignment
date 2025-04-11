# YOLOv5s with Aspect Ratio and Center Alignment Loss

This project modifies the YOLOv5s loss function by adding two additional terms: **Aspect Ratio Loss** and **Center Alignment Loss**. These modifications are integrated into the existing loss function to improve the bounding box predictions for a custom dataset consisting of **cats and dogs**.

The original Yolov5 repository has been copied here for retrieval of functions for evaluation and preprocessing, as will be found in the notebook.

Link to Dataset: https://www.kaggle.com/datasets/andrewmvd/dog-and-cat-detection

## New Loss Function

The new loss function combines the original YOLOv5s loss with the added aspect ratio and center alignment loss terms. Here’s a breakdown of the original and modified loss functions:

### Original YOLOv5s Loss Function

The original YOLOv5s loss function includes the following components:

1. **IoU Loss**: Measures the intersection over union of predicted and ground truth bounding boxes.
2. **Objectness Loss**: Binary cross-entropy loss to predict the presence of an object.
3. **Classification Loss**: Binary cross-entropy loss for classifying objects.

### Modified Loss Function

The modified loss function includes the same components as the original loss function, with the addition of:

1. **Aspect Ratio Loss**: Measures the difference in aspect ratio between the predicted and ground truth bounding boxes to encourage more accurate box dimensions.
   
   ![aspect loss](https://latex.codecogs.com/png.image?\bg_white&space;\dpi{150}&space;\mathcal{L}_{\text{aspect}}%20=%20\left|%20\frac{w_{\text{pred}}}{h_{\text{pred}}}%20-%20\frac{w_{\text{gt}}}{h_{\text{gt}}}%20\right|)

   
2. **Center Alignment Loss**: Calculates the L2 distance between the predicted and ground truth bounding box centers to improve the precision of the object localization.
   
   ![center loss](https://latex.codecogs.com/png.image?\bg_white&space;\dpi{150}&space;\mathcal{L}_{\text{center}}&space;=&space;\|&space;\mathbf{c}_{\text{pred}}&space;-&space;\mathbf{c}_{\text{gt}}&space;\|_2^2)


The final loss function is a weighted sum of the original YOLOv5s loss terms and the added aspect ratio and center alignment loss terms:

![total loss](https://latex.codecogs.com/png.image?\bg_white&space;\dpi{150}&space;\mathcal{L}&space;=&space;\lambda_{\text{IoU}}&space;\mathcal{L}_{\text{IoU}}&space;+&space;\lambda_{\text{center}}&space;\mathcal{L}_{\text{center}}&space;+&space;\lambda_{\text{aspect}}&space;\mathcal{L}_{\text{aspect}}&space;+&space;\lambda_{\text{obj}}&space;\mathcal{L}_{\text{obj}}&space;+&space;\lambda_{\text{BCE}}&space;\mathcal{L}_{\text{BCE}})


Where:

- $\lambda_{\text{IoU}}, \lambda_{\text{center}}, \lambda_{\text{aspect}}, \lambda_{\text{obj}}, \lambda_{\text{BCE}}$ are the weights for each component of the loss function.

## Custom Dataset

This project was trained on a custom dataset consisting of **cats** and **dogs** images. The dataset is used to detect and classify these two object types, and the new loss function was designed to improve the precision of the bounding box predictions.

The dataset consists of images labeled with bounding boxes around the cats and dogs. It is split into training and validation sets to allow the model to generalize to new, unseen data.

## How to Run

Before running the code, make sure to change the home directory to the one where your notebook is located. This happens in the notebook but you should change kaggle to whatever path is your current directory. This applies to the dataset directory paths as well. For example:

```python
import os
os.chdir('/directory_your_notebook_is_in')
```
To run this project, follow these steps:

1. **Install Dependencies**: 
   - Make sure you have Python 3.x installed on your system.
   - Install the required dependencies by running:
   
   ```bash
   pip install -r requirements.txt

The requirements.txt file contains all the dependencies required to run the project, including PyTorch, YOLOv5, and other necessary libraries.

## Train and Validation

After installing the dependencies, simply open and run the **Assignment.ipynb** Jupyter notebook after making the appropriate changes to the paths as mentioned above. It contains all the necessary code to load the custom dataset, load the new and default loss functions, and train the model and validate the model, with the option to test for a given sample case.

## Dataset Structure

The dataset should be in the following structure

project/
│
├── images/
│   └── *.png, *.jpg, etc.
│
├── annotations/
│   └── *.xml


### Folder Descriptions

- **`images/`**  
  This folder should contain all the image files. Each image must have a corresponding annotation file in the `annotations/` folder.

- **`annotations/`**  
  This folder must contain XML annotation files (in PASCAL VOC format) for each image.  
  The filenames of the annotation files must exactly match their corresponding images (e.g., `Cats_Test1.xml` for `Cats_Test1.png`).

### Annotation Format

Each XML file should follow this structure:

```xml
<annotation>
    <folder>images</folder>
    <filename>Cats_Test1.png</filename>
    <size>
        <width>500</width>
        <height>500</height>
        <depth>3</depth>
    </size>
    <segmented>0</segmented>
    <object>
        <name>dog</name>
        <pose>Unspecified</pose>
        <truncated>0</truncated>
        <occluded>0</occluded>
        <difficult>0</difficult>
        <bndbox>
            <xmin>128</xmin>
            <ymin>22</ymin>
            <xmax>240</xmax>
            <ymax>222</ymax>
        </bndbox>
    </object>
</annotation>
