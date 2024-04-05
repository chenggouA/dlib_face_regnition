# 1 导入库
import cv2
import dlib
import numpy as np

# 定义：关键点编码为128D
def encoder_face(image, detector, predictor, encoder, upsample=1, jet=1):
    # 检测人脸
    faces = detector(image, upsample)
    # 对每张人脸进行关键点检测
    faces_keypoints = [ predictor(image, face) for face in faces ] # 每张人脸的关键点
    return [ np.array(encoder.compute_face_descriptor(image, face_keypoint, jet)) for face_keypoint in faces_keypoints ]


# 定义：人脸比较，通过欧氏距离
def compare_faces(face_encoding, test_encoding):
    return list(np.linalg.norm(np.array(face_encoding) - np.array(test_encoding), axis=1))

# 定义：人脸比较，输出对应的名称
def comapre_faces_order(face_encoding, test_encoding, names):
    distance = list(np.linalg.norm(np.array(face_encoding) - np.array(test_encoding), axis=1))
    return zip(*sorted(zip(distance, names)))

import os
def main():

    baseurl = "img"
    # 2 读取4张图片
    img1 = cv2.imread(os.path.join(baseurl, "guo.jpg"))
    img2 = cv2.imread(os.path.join(baseurl, "liu1.jpg"))
    img3 = cv2.imread(os.path.join(baseurl, "liu2.jpg"))
    img4 = cv2.imread(os.path.join(baseurl, "liu3.jpg"))
    test = cv2.imread(os.path.join(baseurl, "liu4.jpg"))
    # BGR to RGB
    img1 = img1[:, :, ::-1]
    img2 = img2[:, :, ::-1]
    img3 = img3[:, :, ::-1]
    img4 = img4[:, :, ::-1]
    test = test[:, :, ::-1]

    img_names = ["guo,jpg", "liu1.jpg", "liu2.jpg", "liu3.jpg"]

    # 3  加载人脸检测器
    detector = dlib.get_frontal_face_detector()
    
    result = detector(img1, 1)



    # 4 加载关键点的检测器
    predictor = dlib.shape_predictor("model/shape_predictor_68_face_landmarks.dat")

    # 5 加载人脸特征编码模型

    # 6 调用方法：128D特征向量输出
    img1_128D = encoder_face(img1, detector, predictor, encoder)[0]
    img2_128D = encoder_face(img2, detector, predictor, encoder)[0]
    img3_128D = encoder_face(img3, detector, predictor, encoder)[0]
    img4_128D = encoder_face(img4, detector, predictor, encoder)[0]
    test_128D = encoder_face(test, detector, predictor, encoder)[0]

    four_images_128D = [img1_128D, img2_128D, img3_128D, img4_128D]

    # 7 调用方法：比较人脸，计算特征向量之间的距离，判断是否为同一人
    distance = compare_faces(four_images_128D, test_128D)
    print(distance)

    distance, name = comapre_faces_order(four_images_128D, test_128D, img_names)

    print("\n")
    print("distance: {}, \n names: {} ".format(distance, name))

if __name__ == '__main__':
    main()