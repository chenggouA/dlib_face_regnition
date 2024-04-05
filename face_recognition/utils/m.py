import dlib
import os
import numpy as np

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("face_recognition/model/shape_predictor_68_face_landmarks.dat")


encoder = dlib.face_recognition_model_v1("face_recognition/model/dlib_face_recognition_resnet_model_v1.dat")

def crop_img(img, result):
    faces = []
    for bbox in result:
        x = bbox.left()
        y = bbox.top()
        w = bbox.width()
        h = bbox.height()
        # 裁剪人脸区域（注意顺序：先y后x）
        faces.append(img[y: y + h, x: x + w])
    return faces

def get_face_img(img, upsample=1):
    result = detector(img, upsample)
    faces = crop_img(img, result)
    return faces, result


def get_faces_keypoints(img, upsample=1):
    faces = detector(img, upsample)

    result = []
    for face in faces:
        pred = predictor(img, face)
        keypoints = []
        for i in range(0, pred.num_parts):
            keypoints.append((pred.part(i).x, pred.part(i).y))

        result.append(keypoints)
    
    return result
    
# 定义：关键点编码为128D
def encoder_face(image, upsample=1, jet=1):

    # 检测人脸
    faces = detector(image, upsample)
    # 对每张人脸进行关键点检测
    faces_keypoints = [ predictor(image, face) for face in faces ] # 每张人脸的关键点
    return [ np.array(encoder.compute_face_descriptor(image, face_keypoint, jet)) for face_keypoint in faces_keypoints ]


    

    