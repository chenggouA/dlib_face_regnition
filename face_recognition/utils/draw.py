import cv2
import dlib

def draw_dlib_bbox_on_image(image, bboxes, color=(0, 255, 0), thickness=2):
    """
    在图像上绘制 dlib 的边界框
    参数:
        image: 要绘制的图像
        bboxes: dlib.rectangles 对象，包含多个边界框
        color: 边界框的颜色, 默认为绿色
        thickness: 边界框线的厚度, 默认为 2px
    返回:
        带有边界框的图像
    """
    for bbox in bboxes:
        # dlib.rectangle 对象有 .left(), .top(), .right(), .bottom() 方法
        left, top, right, bottom = bbox.left(), bbox.top(), bbox.right(), bbox.bottom()
        
        # 绘制边界框
        cv2.rectangle(image, (left, top), (right, bottom), color, thickness)

    return image


def draw_circle_on_image(img, point, color=(255, 0, 0) , radius=3, thickness=-1):
    if isinstance(point, list):
        for i in point:
            draw_circle_on_image(img, i, color, radius, thickness)
    else:
        x, y = point
        cv2.circle(img, (x, y), radius, color, thickness)



def draw_box_name_on_image(frame, name, left, top, right, bottom):

    cv2.rectangle(frame, (left, top), (right, bottom), (0,255,0), 3)
    
    # 17 显示名字
    cv2.putText(frame, name, (left + 10 , bottom-10), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 1)
        
    