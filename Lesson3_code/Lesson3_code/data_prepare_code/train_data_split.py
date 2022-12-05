import argparse
import xml.etree.ElementTree as ET
from utilis import *
import argparse

label_list = ['person']

def get_image_txt(opt):

    #　阶段一：对于数据集进行清洗梳理　
    # 第一步：根据images_label_split中的图像删除多余的xml
    # print("V1")
    # compare_image_label_remove_xml(opt.train_data)
    # # # 第二步：根据images_label_split中的图像删除多余的image
    # print("V2")
    # compare_image_label_remove_image(opt.train_data)
    # # 第三步：将各个文件夹中的xml不满足条件的文件删除
    # print("V3")
    # remove_not_satisfied_xml(opt.train_data)
    # # 第四步：查找xml是否为空，空的话删除xml,也删除对应的image
    # print("V4")
    # remove_image_null_xml(opt.train_data,label_list)
    # # 第五步：对照image和xml中数据，显示图片看画得框是否正确
    # show_label(opt.train_data,label_list)

    #　阶段二：将数据按照一定比例分成训练和验证集　
    # 将train和test随机分开，将image和xml分别保存到train和test所在的文件夹中
    # 根据前面可以得到xml和image,每个场景下选择10%的数据,作为验证集, 生成train和test两个文件夹
    # yolov3_get_train_test_file(opt.train_data,0.2)

    # 阶段三：将train和test的xml，转换成txt
    # 第一步：将train和test中的xml文件生成txt文件，都放到image_txt文件夹中
    yolov3_get_txt(opt.train_data,label_list)
    # #  第二步：将所有的image文件一起移动到image_txt中
    yolov3_move_image(opt.train_data)
    # # 第三步：将train/Annotations和test/Annotations的xml自动生成train.txt和test.txt文件，并保存到train_test_txt中
    yolov3_get_train_test_txt(opt.train_data)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_data', type=str, default='D:/aidlux/lesson/Lesson3/lesson3_data/train_data', help='data dir')
    opt = parser.parse_args()
    get_image_txt(opt)