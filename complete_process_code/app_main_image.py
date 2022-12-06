# aidlux相关
from cvs import *
import aidlite_gpu
from yolov5_code.aidlux.utils import \
    detect_postprocess, preprocess_img, draw_detect_res, extract_detect_res

import requests
import time
import cv2

import matplotlib.pyplot as plt
import os
from skimage.io import imread
import torch
import torch.nn as nn
import torchvision.utils
from torchvision.models import mobilenet_v2,resnet18
from adv_code.advertorch.utils import predict_from_logits
from adv_code.advertorch.utils import NormalizeByChannelMeanStd
from adv_code.robust_layer import GradientConcealment, ResizedPaddingLayer
from adv_code.timm.models import create_model

from adv_code.advertorch.attacks import LinfPGDAttack, FGSM
from adv_code.advertorch_examples.utils import ImageNetClassNameLookup
from adv_code.advertorch_examples.utils import bhwc2bchw
from adv_code.advertorch_examples.utils import bchw2bhwc

from pathlib import Path

from adv_code.detect_adv_code import Model, Detect_Model

device = "cuda" if torch.cuda.is_available() else "cpu"

def extract_car_img(source, images_lst):
    # 读取图片进行推理
    frame_id = 0
    frame_lst, pred_lst = [], []
    for image_name in images_lst:
        frame_id += 1
        print("frame_id:", frame_id)
        image_path = os.path.join(source, image_name)
        # 读取数据集
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

        frame_lst.append(frame)
        pred_lst.append(pred)
        # cvs.imshow(res_img)

        # 测试结果展示停顿
        #time.sleep(5)

        # cvs.imshow(cut_img)
        # cap.release()
        # cv2.destroyAllWindows()

    return frame_lst, pred_lst, images_lst


def tensor2npimg(tensor):
    return bchw2bhwc(tensor[0].cpu().numpy())


### 展示攻击结果
def show_images(model, img, advimg, img_name=None, enhance=127):
    np_advimg = tensor2npimg(advimg)
    np_perturb = tensor2npimg(advimg - img)

    pred = imagenet_label2classname(predict_from_logits(model(img)))
    advpred = imagenet_label2classname(predict_from_logits(model(advimg)))

    

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 3, 1)
    plt.imshow(np_img)

    plt.axis("off")
    plt.title("original image\n prediction: {}".format(pred))
    plt.subplot(1, 3, 2)
    plt.imshow(np_perturb * enhance + 0.5)

    plt.axis("off")
    plt.title("the perturbation,\n enhanced {} times".format(enhance))
    plt.subplot(1, 3, 3)
    plt.imshow(np_advimg)
    plt.axis("off")
    plt.title("perturbed image\n prediction: {}".format(advpred))
    
    plt.show()
    # plt.savefig(f'/home/intelligent_transportation/Lesson5_code/adv_code/{img_name}')


### 读取图片
def get_image(img_path):
    img_url = "https://farm1.static.flickr.com/230/524562325_fb0a11d1e1.jpg"

    def _load_image():
        return imread(img_path) / 255.

    if os.path.exists(img_path):
        return _load_image()
    else:
        print(f'{img_path}不存在...')
        exit()
        import urllib
        urllib.request.urlretrieve(img_url, img_path)
        return _load_image()


def get_advimg(img_path, is_show=False):
    ### 数据预处理
    np_img = get_image(img_path=img_path)
    img = torch.tensor(bhwc2bchw(np_img))[None, :, :, :].float().to(device)
    imagenet_label2classname = ImageNetClassNameLookup()

    ### 测试模型输出结果
    pred = imagenet_label2classname(predict_from_logits(model(img)))
    print("test output:", pred)

    ### 输出原label
    pred_label = predict_from_logits(model_su(img))  
    # 使用替身模型的相关信息进行对抗样本的获取
    # 这包括替身模型的参数、输出label

    ### 完成攻击，输出对抗样本
    advimg = adversary.perturb(img, pred_label)

    ### 展示源图片，对抗扰动，对抗样本以及模型的输出结果
    if is_show:
        show_images(model, img, advimg, img_name=f'show_{int(i)}_{pred}.png')

    ### 迁移攻击样本保存
    save_path = "/home/intelligent_transportation/Lesson5_code/adv_code/adv_results/"
    torchvision.utils.save_image(advimg.cpu().data, save_path + f"adv_image_{int(i)}_{pred}.png")
    
    return advimg, save_path + f"adv_image_{int(i)}_{pred}.png"


if __name__ == '__main__':
    # 车辆检测

    # # AidLite初始化：调用AidLite进行AI模型的加载与推理，需导入aidlite
    # aidlite = aidlite_gpu.aidlite()
    # # Aidlite模型路径
    # model_path = '/home/intelligent_transportation/Lesson5_code/yolov5_code/aidlux/yolov5_car_best-fp16.tflite'
    # # 定义输入输出shape
    # in_shape = [1 * 640 * 640 * 3 * 4]
    # out_shape = [1 * 25200 * 6 * 4]
    # # 加载Aidlite检测模型：支持tflite, tnn, mnn, ms, nb格式的模型加载
    # aidlite.ANNModel(model_path, in_shape, out_shape, 4, 0)
    
    # # imgs_path = "/home/intelligent_transportation/Lesson5_code/yolov5_code/data/images/tests"
    # # imgs_path = '/home/intelligent_transportation/AI_test_data/test_adv'
    
    # # 设置测试集路径
    # images_lst = os.listdir(imgs_path)
    # images_lst = [p for p in images_lst if p.endswith('jpg') or p.endswith('png')]
    # print(images_lst)
    # # extract_res: frame_lst, pred_lst, images_lst
    # extract_res = extract_car_img(imgs_path, images_lst)


    # # 检测框提取

    # save_folder = "/home/intelligent_transportation/Lesson5_code/yolov5_code/aidlux/extract_results/"
    # # 图片裁剪，提取车辆目标区域
    # car_img_name_lst = []
    # for frame, pred, img_name in zip(*extract_res):
    #     img_name_lst = extract_detect_res(frame, pred, img_name, save_folder)
    #     car_img_name_lst.extend(img_name_lst)
    # assert car_img_name_lst != [], '没有提取到车辆框！'


    # # 使用对抗样本

    # ### 替身模型加载
    # normalize = NormalizeByChannelMeanStd(
    #     mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    # # model_su = resnet18(pretrained=True)
    # model_su = resnet18(pretrained=False)
    # resnet18_param = torch.load('/home/intelligent_transportation/Lesson5_code/model/resnet18-5c106cde.pth')
    # model_su.load_state_dict(resnet18_param)
    # model_su.eval()
    # model_su = nn.Sequential(normalize, model_su)
    # model_su = model_su.to(device)

    # ### 对抗攻击：PGD攻击算法
    # # adversary = LinfPGDAttack(
    # #    model_su, eps=8/255, eps_iter=2/255, nb_iter=80,
    # #    rand_init=True, targeted=False)
    # adversary = FGSM(
    #     model_su, eps=8/255, targeted=False
    # )   
    
    # use_advimg = False
    # print(f'是否使用对抗样本: {use_advimg}...')
    # print('车辆框信息：', car_img_name_lst)
    # img_res = []
    # for i, img_name in enumerate(car_img_name_lst):
    #     print(img_name)
    #     img_path = save_folder + img_name
    #     if use_advimg:
    #         advimg, advimg_path = get_advimg(img_path)
    #         img_res.append(advimg)
    #     else:
    #         np_img = get_image(img_path)
    #         img = torch.tensor(bhwc2bchw(np_img))[None, :, :, :].float().to(device)
    #         img_res.append(img)
    
    # AI安全监测与告警
    model = Model().eval().to(device)
    detect_model = Detect_Model().eval().to(device)
    imagenet_label2classname = ImageNetClassNameLookup()

    imgs_path = Path('/home/intelligent_transportation/AI_test_data/test')
    img_path_lst = [p for p in imgs_path.rglob('*') 
            if p.name.endswith('jpg') or p.name.endswith('png')]
    for img_path in img_path_lst:
        np_img = get_image(img_path)
        img = torch.tensor(bhwc2bchw(np_img))[None, :, :, :].float().to(device)
        ### 对抗攻击监测
        detect_pred = detect_model(img)
        print('=' * 60)
        print(detect_pred)

        cvs.imshow(np_img * 255.)
        
        if detect_pred > 0.5:
            id = 'tHyzzn1'
            # 填写喵提醒中，发送的消息，这里放上前面提到的图片外链
            text = "出现对抗攻击风险！！"
            ts = str(time.time())  # 时间戳
            type = 'json'  # 返回内容格式
            request_url = "http://miaotixing.com/trigger?"

            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/87.0.4280.67 Safari/537.36 Edg/87.0.664.47'}

            result = requests.post(request_url + "id=" + id + "&text=" + text + "&ts=" + ts + "&type=" + type,
                                headers=headers)
            print('存在对抗攻击风险！')
        else:
            pred = imagenet_label2classname(predict_from_logits(model(img)))
            print('result: ', pred)




