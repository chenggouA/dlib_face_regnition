from .utils import *
import numpy as np
import os


class Base:

    def __init__(self):
        self.names = None
        self.encodings = None

        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor("face_recognition/model/shape_predictor_68_face_landmarks.dat")


        self.encoder = dlib.face_recognition_model_v1("face_recognition/model/dlib_face_recognition_resnet_model_v1.dat")
        self.read_data()

    def read_data(self):
        raise NotImplementedError("子类必须实现此方法")
    
    def inference(self, ori_img):
        raise NotImplementedError("子类必须实现此方法")
    
    

    def crop_img(self, img, result):
        faces = []
        for bbox in result:
            x = bbox.left()
            y = bbox.top()
            w = bbox.width()
            h = bbox.height()
            # 裁剪人脸区域（注意顺序：先y后x）
            faces.append(img[y: y + h, x: x + w])
        return faces

    def get_face_img(self, img, upsample=1):
        result = self.detector(img, upsample)
        faces = self.crop_img(img, result)
        return faces, result


    def get_faces_keypoints(self, img, upsample=1):
        faces = self.detector(img, upsample)

        result = []
        for face in faces:
            pred = self.predictor(img, face)
            keypoints = []
            for i in range(0, pred.num_parts):
                keypoints.append((pred.part(i).x, pred.part(i).y))

            result.append(keypoints)
        
        return result
    
    def compare_faces(self, encodings, face_encoding, threshold = 0.4):
        # 
        distances = self.face_distance(encodings, face_encoding)
        matches = distances <= threshold
        return matches.tolist()
    
    def face_distance(self, encodings, face_encoding):
        return np.linalg.norm(np.array(encodings) - face_encoding, axis = 1)

        
    # 定义：关键点编码为128D
    def encoder_face(self, image, faces, jet=1):

        # 对每张人脸进行关键点检测
        faces_keypoints = [ self.predictor(image, face) for face in faces ] # 每张人脸的关键点
        return [ np.array(self.encoder.compute_face_descriptor(image, face_keypoint, jet)) for face_keypoint in faces_keypoints ]
    
class FaceRecognitionFromArray(Base):

    def __init__(self, save_dir):
        self.save_dir = save_dir
        super().__init__()

    def read_data(self):
        self.names = np.load(os.path.join(self.save_dir, "names.npy"))
        self.encodings = np.load(os.path.join(self.save_dir, "encodings.npy"))

    def inference(self, ori_img):

        frame = ori_img.copy()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # 检测人脸
        faces = self.detector(frame, 1)

        encodings = self.encoder_face(frame, faces)

        for encoding, face in zip(encodings, faces):

            left = face.left()
            top = face.top()
            bottom = face.bottom()
            right = face.right()

            name = "Unknown"

            if len(self.encodings) != 0:

                matches = self.compare_faces(self.encodings, encoding)

                distances = self.face_distance(self.encodings, encoding)

                min_distance_index = np.argmin(distances)


                if matches[min_distance_index]:
                    name = self.names[min_distance_index]

            draw_box_name_on_image(ori_img, name, left, top, right, bottom)
        
        
        return ori_img