# SSD_mobilenetv2-with-Focal-loss

 this project is focked from https://github.com/amdegroot/ssd.pytorch.

add functions:
1. implement mobielentv2 for ssd.
2. add focal loss. (need adjuct super-parameters)
3. add detection.py demo for image and video detection.


result(train on voc 2007trainval + 2012, test on voc 2007test): 
ssd-mobielnetv2(this repo): 70.27%(without focal loss)
ssd-vgg16(paper):77.20%
ssd-mobielentv1: 72.7%(with COCO pretraining), 68.%(without COCO pretaining)  https://github.com/chuanqi305/MobileNet-SSD\n

pretrained model and trained model: 
百度网盘: https://pan.baidu.com/s/1RmOPF4jQYpYlE_8E4DifeQ 提取码: f53n 
