from face_recognition.model import FaceRecognitionFromArray
import cv2

save_dir = "./encoding"
model = FaceRecognitionFromArray(save_dir)



img = cv2.imread("liu4.jpg")

img = model.inference(img)

cv2.imshow("img", img)
cv2.waitKey(0)