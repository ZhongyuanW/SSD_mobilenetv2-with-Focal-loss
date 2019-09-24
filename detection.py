import os
import torch
from torch.autograd import Variable
import numpy as np
import cv2
import matplotlib
from data import VOC_CLASSES as labels
import time
import data.config as cfg


from ssd_mobilenetv2 import build_ssd

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

if torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')

num_classes = 21
top_k=20

image_size = 300


def detection_image(path,weight):
    # cv2.namedWindow("result", 0)
    global image_size

    net = build_ssd('test', 300, num_classes)
    net.eval()
    net.load_weights(weight)

    image = cv2.imread(path, cv2.IMREAD_COLOR)

    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # rgb_image = cv2.resize(rgb_image, (512, 512))

    resize_image = cv2.resize(image, (300, 300)).astype(np.float32)
    resize_image -= (104, 117, 123)
    resize_image = resize_image.astype(np.float32)
    resize_image = resize_image[:, :, ::-1].copy()

    torch_image = torch.from_numpy(resize_image).permute(2, 0, 1)

    input_image = Variable(torch_image.unsqueeze(0))
    if torch.cuda.is_available():
        input_image = input_image.cuda()

    out = net(input_image)

    colors = cfg.COLORS

    detections = out.data

    scale = torch.Tensor(rgb_image.shape[1::-1]).repeat(2)

    rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)

    idx_obj = -1

    for i in range(detections.size(1)):
        j = 0
        while detections[0, i, j, 0] >= 0.45:

            idx_obj += 1

            score = detections[0, i, j, 0]
            label_name = labels[i - 1]
            display_txt = '%s %.2f' % (label_name, score)
            pt = (detections[0, i, j, 1:] * scale).cpu().numpy()

            j += 1

            pt[0] = max(pt[0], 0)
            pt[1] = max(pt[1], 0)
            pt[2] = min(pt[2], rgb_image.shape[0])
            pt[3] = min(pt[3], rgb_image.shape[1])

            color = colors[idx_obj%(len(colors))]

            textsize = cv2.getTextSize(display_txt, cv2.FONT_HERSHEY_COMPLEX, 1, 2)[0]

            text_x = int(pt[0])
            text_y = int(pt[1])
            if (int(pt[1]) - textsize[1] < 0):
                text_y = int(pt[1]) + textsize[1] + 2
                cv2.rectangle(rgb_image, (int(pt[0]), int(pt[1])),
                              (int(pt[0]) + textsize[0] + 8, int(pt[1]) + textsize[1] + 10),
                              (color[0], color[1], color[2], 125), -1)
            else:
                text_y -= 6
                cv2.rectangle(rgb_image, (int(pt[0]) - 2, int(pt[1]) - textsize[1] - 10),
                              (int(pt[0]) + textsize[0] + 8, int(pt[1])), (color[0], color[1], color[2], 125), -1)

            cv2.rectangle(rgb_image, (int(pt[0]), int(pt[1])), (int(pt[2]), int(pt[3])), color, 4)
            cv2.putText(rgb_image, display_txt, (text_x + 4, text_y), cv2.FONT_HERSHEY_COMPLEX, 1,
                        (255 - color[0], 255 - color[1], 255 - color[2]), 2)
            cv2.putText(rgb_image, 'x', (int((pt[2] + pt[0]) // 2 - 5), int((pt[3] + pt[1]) // 2)),
                        cv2.FONT_HERSHEY_COMPLEX, 1, color)

        # cv2.imshow("result", rgb_image)
        # cv2.waitKey(0)
    print(path.replace("test_images", "out_images"))
    cv2.imwrite(path.replace("test_images","out_images"),rgb_image)


def detection_video(path,weight):
    global image_size

    flag = 0
    net = build_ssd('test', 300, num_classes)
    net.eval()
    net.load_weights(weight)

    cap = cv2.VideoCapture(path)
    frameNumber = cap.get(7)

    fps = cap.get(cv2.CAP_PROP_FPS)
    size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    fourcc = cv2.VideoWriter_fourcc('M', 'P', '4', '2')
    outVideo = cv2.VideoWriter('output_videos/out_%s.avi'%(path.split("/")[-1].split(".")[0]), fourcc, fps, size)
    cv2.namedWindow("result", 0)

    # image_size = size[0]

    while cap.isOpened():
        ret,image = cap.read()
        flag += 1

        if ret == False:
            print("video is over!")
            break
        if flag % 3 != 0:
            continue

        t0 = time.time()
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        #rgb_image = cv2.resize(rgb_image, (512, 512))

        resize_image = cv2.resize(image, (300, 300)).astype(np.float32)
        resize_image -= (104, 117, 123)
        resize_image = resize_image.astype(np.float32)
        resize_image = resize_image[:, :, ::-1].copy()

        torch_image = torch.from_numpy(resize_image).permute(2, 0, 1)

        input_image = Variable(torch_image.unsqueeze(0))
        if torch.cuda.is_available():
            input_image = input_image.cuda()

        out = net(input_image)

        colors = cfg.COLORS

        detections = out.data

        scale = torch.Tensor(rgb_image.shape[1::-1]).repeat(2)


        rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)

        idx_obj = -1

        for i in range(detections.size(1)):
            j = 0
            while detections[0,i,j,0] >= 0.45:

                idx_obj += 1

                score = detections[0,i,j,0]
                label_name = labels[i-1]

                display_txt = '%s %.2f'%(label_name, score)
                pt = (detections[0,i,j,1:]*scale).cpu().numpy()

                j += 1

                # TODO revise solutions
                pt[0] = max(pt[0],0)
                pt[1] = max(pt[1],0)
                pt[2] = min(pt[2],size[0])
                pt[3] = min(pt[3],size[1])


                color = colors[idx_obj%len(colors)]

                textsize = cv2.getTextSize(display_txt, cv2.FONT_HERSHEY_COMPLEX, 1, 2)[0]


                text_x = int(pt[0])
                text_y = int(pt[1])
                if (int(pt[1])-textsize[1]<0):
                    text_y = int(pt[1]) + textsize[1] + 2
                    cv2.rectangle(rgb_image,(int(pt[0]),int(pt[1])),(int(pt[0])+textsize[0]+8,int(pt[1])+textsize[1]+10),(color[0],color[1],color[2],125),-1)
                else:
                    text_y -= 6
                    cv2.rectangle(rgb_image, (int(pt[0])-2, int(pt[1])-textsize[1]-10),(int(pt[0]) + textsize[0]+8, int(pt[1])), (color[0],color[1],color[2],125), -1)


                cv2.rectangle(rgb_image,(int(pt[0]), int(pt[1])),(int(pt[2]), int(pt[3])),color,4)
                cv2.putText(rgb_image, display_txt, (text_x + 4, text_y), cv2.FONT_HERSHEY_COMPLEX, 1,(255 - color[0], 255 - color[1], 255 - color[2]), 2)

        t1 = time.time()

        cv2.putText(rgb_image, "FPS: %.2f" % (1 / (t1 - t0)), (5, 30), cv2.FONT_HERSHEY_COMPLEX, 1.2, (255, 255, 255), 2)

        # cv2.imshow("result",rgb_image)
        outVideo.write(rgb_image)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            outVideo.release()
            cap.release()
            cv2.destroyAllWindows()


def detection(path,weight):
    if path[-3:] == "jpg":
        if os.path.isfile(path):
            detection_image(path,weight)
        else:
            print("not finding %s"%(path))
    elif path[-3:] == "avi":
        if os.path.isfile(path):
            detection_video(path,weight)
        else:
            print("not finding %s"%(path))
    else:
        print("format is %s\n"%path[-3:])
        print("%s is not a image or video!"%path)

import os

if __name__ == "__main__":
    weight = 'weights/ssd_mobilenetv2_300/mobilenetv2_final.pth'
    path = r"test_images/2007_000256.jpg"
    detection(path,weight)

