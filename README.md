# SSD_mobilenetv2-with-Focal-loss

 this repo is forked from https://github.com/amdegroot/ssd.pytorch. Implement by pytorch.

add functions:
1. implement mobielentv2 for ssd.
2. add focal loss. (need adjuct super-parameters).
3. add detection.py demo for image and video detection.


result(train on voc 2007trainval + 2012, test on voc 2007test):
1. ssd-mobielnetv2 (this repo): 70.27%. (without focal loss).
2. ssd-mobielentv1: 68.% (without COCO pretaining), 72.7% (with COCO pretraining)   https://github.com/chuanqi305/MobileNet-SSD.
3. ssd-vgg16 (paper): 77.20%. 

pretrained model and trained model: 
1. 百度网盘: https://pan.baidu.com/s/1RmOPF4jQYpYlE_8E4DifeQ 提取码: f53n 
2. Google drive: https://drive.google.com/drive/folders/1JoDYukyWZZ-iWVWPhUDD998cSPB3LeUw?usp=sharing
