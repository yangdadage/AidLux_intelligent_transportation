# aidlux相关
from cvs import *
import aidlite_gpu
from utils import detect_postprocess, preprocess_img, draw_detect_res, extract_detect_res

import time
import cv2

# AidLite初始化：调用AidLite进行AI模型的加载与推理，需导入aidlite
aidlite = aidlite_gpu.aidlite()
# Aidlite模型路径
model_path = '/home/Lesson5_code/yolov5_code/models/yolov5_car_best-fp16.tflite'
# 定义输入输出shape
in_shape = [1 * 640 * 640 * 3 * 4]
out_shape = [1 * 25200 * 6 * 4]
# 加载Aidlite检测模型：支持tflite, tnn, mnn, ms, nb格式的模型加载
aidlite.ANNModel(model_path, in_shape, out_shape, 4, 0)

# 读取图片进行推理
# 设置测试集路径
source = "/home/Lesson5_code/yolov5_code/data/images/tests"
images_list = os.listdir(source)
print(images_list)
frame_id = 0
# 读取数据集
for image_name in images_list:
    frame_id += 1
    print("frame_id:", frame_id)
    image_path = os.path.join(source, image_name)
    frame = cvs.imread(image_path)

    # 预处理
    img = preprocess_img(frame, target_shape=(640, 640), div_num=255, means=None, stds=None)
    # 数据转换：因为setTensor_Fp32()需要的是float32类型的数据，所以送入的input的数据需为float32,大多数的开发者都会忘记将图像的数据类型转换为float32
    aidlite.setInput_Float32(img, 640, 640)
    # 模型推理API
    aidlite.invoke()
    # 读取返回的结果
    pred = aidlite.getOutput_Float32(0)
    # 数据维度转换
    pred = pred.reshape(1, 25200, 6)[0]
    # 模型推理后处理
    pred = detect_postprocess(pred, frame.shape, [640, 640, 3], conf_thres=0.25, iou_thres=0.45)
    # 绘制推理结果
    res_img = draw_detect_res(frame, pred)
    # cvs.imshow(res_img)

    # 测试结果展示停顿
    #time.sleep(5)

    # 图片裁剪，提取车辆目标区域
    extract_detect_res(frame, pred, image_name)

    # cvs.imshow(cut_img)
    # cap.release()
    # cv2.destroyAllWindows()


