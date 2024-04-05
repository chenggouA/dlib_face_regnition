# 1 加载库
import cv2
import numpy as np
import face_recognition

# 2 加载图片
liu = cv2.imread("liu.jpeg")
guo = cv2.imread("guo.jpg")

# 3 BGR 转 RGB
liu_RGB = liu[:, :, ::-1]
guo_RGB = guo[:, :, ::-1]

# 4 检测人脸
liu_face = face_recognition.face_locations(liu_RGB)
guo_face = face_recognition.face_locations(guo_RGB)

# 5 人脸特征编码
liu_encoding = face_recognition.face_encodings(liu_RGB, liu_face)[0]
guo_encoding = face_recognition.face_encodings(guo_RGB, guo_face)[0]

# 6 把所有人脸放在一起，当做数据库使用
encodings = [liu_encoding, guo_encoding]
names = ["liu de hua", "guo fu cheng"]

# 7 打开摄像头，读取视频流
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise  IOError("Camera Error !")

while True:
    ret, frame = cap.read()
    frame = cv2.resize(frame, (0,0), fx=0.5, fy=0.5)
    # 8 BGR 传 RGB
    frame_RGB = frame[:, :, ::-1]
    # 9 人脸检测
    faces_locations = face_recognition.face_locations(frame_RGB)
    # 10 人脸特征编码
    faces_encodings = face_recognition.face_encodings(frame_RGB, faces_locations)
    # 11 与数据库中的所有人脸进行匹配
    for (top, right, bottom, left), face_encoding in zip(faces_locations, faces_encodings):
        # 12 进行匹配
        matches = face_recognition.compare_faces(encodings, face_encoding)
        # 13 计算距离
        distances = face_recognition.face_distance(encodings, face_encoding)
        min_distance_index = np.argmin(distances) # 0, 1, 2
        # 14 判断：如果匹配，获取名字
        name = "Unknown"
        if matches[min_distance_index]:
            name = names[min_distance_index]
        # 15 绘制人脸矩形框
        
    # 18 显示整个效果
    cv2.imshow("face recognition", frame)
    # 19 判断 Q , 退出
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
# 20 关闭所有资源
cap.release()
cv2.destroyAllWindows()