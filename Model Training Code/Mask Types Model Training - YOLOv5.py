# -*- coding: utf-8 -*-
"""
Mia Flo - Train-YOLOv5

Install Dependencies
DataSets must be from Roboflow
This is Google Colab Format, hindi po recommended sa Python IDE
Para sa compiling purposes only

Piliin ang GPU in Runtime 
Go to Runtime--> Change Runtime Type --> Hardware accelerator --> GPU

"""

# Commented out IPython magic to ensure Python compatibility.
# clone YOLOv5 repository
!git clone https://github.com/ultralytics/yolov5  # clone repo
# %cd yolov5
!git reset --hard 886f1c03d839575afecb059accf74296fad395b6

# install dependencies as necessary
!pip install -qr requirements.txt  # install dependencies (ignore errors)
import torch

from IPython.display import Image, clear_output  # to display images
from utils.google_utils import gdrive_download  # to download models/datasets

# clear_output()
print('Setup complete. Using torch %s %s' % (torch.__version__, torch.cuda.get_device_properties(0) if torch.cuda.is_available() else 'CPU'))

#INSTALL ROBOFLOW
!pip install roboflow

from roboflow import Roboflow
#Sa API Key, provided po ito na code from Roboflow

#Replace lang po ang *your code*
rf = Roboflow(api_key="your code")
#Replace DataSet Title
project = rf.workspace().project("DataSet Title")
dataset = project.version(1).download("yolov5")


"""Define Model Configuration and Architecture
"""

# define number of classes based on YAML
import yaml
with open(dataset.location + "/data.yaml", 'r') as stream:
    num_classes = str(yaml.safe_load(stream)['nc'])

#customize iPython writefile so we can write variables
from IPython.core.magic import register_line_cell_magic

@register_line_cell_magic
def writetemplate(line, cell):
    with open(line, 'w') as f:
        f.write(cell.format(**globals()))

### MODEL TRAINING


Here, we are able to pass a number of arguments:
- **img:** define input image size
- **batch:** determine batch size
- **epochs:** define the number of training epochs. (Note: often, 3000+ are common here!)
- **data:** set the path to our yaml file
- **cfg:** specify our model configuration
- **weights:** specify a custom path to weights. (Note: you can download weights from the Ultralytics Google Drive [folder](https://drive.google.com/open?id=1Drs_Aiu7xx6S-ix95f9kNsA6ueKRpN2J))
- **name:** result names
- **nosave:** only save the final checkpoint
- **cache:** cache images for faster training
"""

# Commented out IPython magic to ensure Python compatibility.
# # train yolov5s on custom data for 100 epochs
# # time its performance
# %%time
# %cd /content/yolov5/
# !python train.py --img 416 --batch 16 --epochs 100 --data {dataset.location}/data.yaml --cfg ./models/custom_yolov5s.yaml --weights '' --name yolov5s_results  --cache

"""
# Evaluate Custom YOLOv5 Detector Performance

#Training losses and performance metrics are saved to Tensorboard and also to a logfile defined above with the **--name** flag when we train.
#In our case, we named this `yolov5s_results`. (If given no name, it defaults to `results.txt`.) The results file is plotted as a png after training completes.

#Note: Partially completed `results.txt` files can be plotted with `from utils.utils import plot_results; plot_results()`.

from google.colab import drive
drive.mount('/content/drive')

from utils.plots import plot_results  # plot results.txt as results.png
from google.colab.patches import cv2_imshow

img = cv2.imread(filename='/content/yolov5/runs/train/yolov5s_results/results.png')  # view results.png
cv2_imshow(img)
cv2.waitKey(0)


### Visualize Our Training Data with Labels

After training starts, view `train*.jpg` images to see training images, labels and augmentation effects.

Note a mosaic dataloader is used for training (shown below), a new dataloading concept developed by Glenn Jocher and first featured in [YOLOv4](https://arxiv.org/abs/2004.10934).


# display our ground truth data
print("GROUND TRUTH TRAINING DATA:")
import cv2
from google.colab.patches import cv2_imshow
img1 = cv2.imread(filename='/content/yolov5/runs/train/yolov5s_results/test_batch0_labels.jpg')
cv2_imshow(img1)
cv2.waitKey(0)

# print out an augmented training example
print("GROUND TRUTH AUGMENTED TRAINING DATA:")

import cv2
from google.colab.patches import cv2_imshow
img3 = cv2.imread(filename='/content/yolov5/runs/train/yolov5s_results/train_batch0.jpg')

cv2_imshow(img3)
cv2.waitKey(0)


#Run Inference  With Trained Weights
#Run inference with a pretrained checkpoint on contents of `test/images` folder downloaded from Roboflow.


#display inference on ALL test images
#this looks much better with longer training above

import glob
from IPython.display import Image, display

for imageName in glob.glob('/content/yolov5/runs/detect/exp/*.jpg'): #assuming JPG
    display(Image(filename=imageName))
    print("\n")

#   Weights Export

#Para ma connect ang training files to your Google Drive

from google.colab import drive
drive.mount('/content/gdrive')


