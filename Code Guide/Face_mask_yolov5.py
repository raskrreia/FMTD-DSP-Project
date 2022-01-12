# first box
from distutils.dir_util import copy_tree
import os

os.mkdir('labels')
os.mkdir('images')
copy_tree('../input/face-mask-detection/images'. './images')

# second box
# Xml parser (extract data from xml file)

import xml.etrr.ElementTree as ET
import os

annotations = ''../input/face-mask-detection/annotations'
for file_ in os.listdir(annotations):
    path = os.path.join(annotations, file_)

    tree = ET.parse(path)
    root = tree.getroot()
    file = root[1].text

    width = int(root[2][0].text)
    height = int(root[2][1].text)

    classes = {'without_mask': 0, 'mask_weared_incorrect': 1, 'with mask': 2}


    lines = ''
    for i, child in enumerate(root):
        if child.tag == 'object':
            class_ = root[i][0].text
              x1, y1, x2, y2 = int(root[i][5][0].text), int(root[i][5][1].text), int(root[i][5][2].text), # putol ni siya diria dili niya iisdog hahahah

              x, y = (((x2 - x1) / 2) + x1) / width, (((y2 - y1) / 2) + y1) / height
              w, h = (x2 - x1) / width, (y2 - y1) / height
              line = f'{classes[class_]} {x} {y} {w} {h} \n'

    with open(f"labels/{file.replace('.png", '.txt).replace('.jpeg', '.txt').replace('.jpg', '.txt')}", # putol  sad ni siya diria dili niya iisdog hahahah
        f.write(lines)


# third box
os.listdir('./labels')    

# fourth box
!git clone https://github.com/ultalytics/yolov5
%cd yolov5
%pip install -qr requirements.txt #install dependencies

import torch
from IPythin.display import Image, clear_output # t display images

clear_output()
print(f"Setup complete. Using torch {torch.__version__} ({torch.cuda.get_device_properties(0).name if  torch.cuda.is_available() else 'CPU'})")

# fifth box
%pip install -q wandib
!wandib disabled

# sixth box
# !cat data/coco128.yaml

# seventh box
data = """
path: ../ #data root dir
train: images # train images (relative to 'path') 128 images
val: images #val images (relative to 'path') 128 images

# Classes
nc: 3 # number of classes
names: ['without mask', 'incorrect mask' 'with mask'] #class names

"""


with open('data/custom_data.yaml', 'w') as w:
    w.write(data)

# eighth box
!python train.py --img 640 --epochs 50 --data custom.data.yaml --weights yolov5s.pt --cache

# ninth box
from shutil import copyfile
copyfile('_____', '../best.pt')