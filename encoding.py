import dlib
import os
import cv2
from face_recognition.utils import draw_dlib_bbox_on_image
from face_recognition.utils import encoder_face, image_check
import numpy as np
dataset = "face_recognition/img"
encodings = []
names = []
save_dir = "./encoding"
for img in os.listdir(dataset):
    
    if not image_check(img):
        continue
    
    # 通过下划线区分名字
    name = img.split(".")[0].split("_")[0]
    names.append(name)
    img_path = os.path.join(dataset, img)
    
    im = cv2.imread(img_path)

    im_rgb = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

    # 数据图片中应只有一张人脸
    encoding = encoder_face(im_rgb)[0]
    

    encodings.append(encoding)

    # cv2.imshow("img", im)
    # cv2.waitKey(0)

os.makedirs(save_dir, exist_ok=True)
np.save(os.path.join(save_dir, "encodings"), encodings)
np.save(os.path.join(save_dir, "names"), names)

    