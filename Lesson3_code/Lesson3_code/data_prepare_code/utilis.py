# from Dataset_Utlize.my_tools.Scratch import path_remake
import os
from shutil import copyfile
import xml.etree.ElementTree as ET
import cv2
import random
from tqdm import tqdm
import sys

def path_remake(path):
    return path.replace(' ', '\ ').replace('(','\(').replace(')','\)')

img_format = ['.jpg', '.jpeg','.png']
xml_format = ['.xml']

colors_list = [(139, 0, 139), 
               (238, 104, 123), 
               (160, 158, 95),
               (32, 165, 218), 
               (50, 250, 50), 
               (255.0,0),
               (0, 255, 0),
               (0,0,255), 
               (30,105,210),
               (100,226,127)]


# 将图片和result的分离开来
def images_results_split(total_dir,split_dir):
    totalList = os.listdir(total_dir)
    count = 0
    for filename in totalList:
        #判断image_result_split里面相关文件在不在
        # split_file_name = split_dir+filename
        split_file_name = os.path.join(split_dir, filename)

        # split_image_name = split_file_name + "/images"
        split_image_name = os.path.join(split_file_name, 'images')

        # split_result_name = split_file_name + "/results"
        split_result_name = os.path.join(split_file_name, 'results')

        isExists = os.path.exists(split_file_name)
        # 判断是否存在文件夹，不存在则进行新建
        if not isExists:
            os.makedirs(split_file_name)
            os.makedirs(split_image_name)
            os.makedirs(split_result_name)
            print(str(split_file_name) + "新建成功")
        # 将原始文件复制到目标文件中
        # total_path = total_dir+filename
        total_path = os.path.join(total_dir, filename)
        totalList = os.listdir(total_path)
        for file_name in totalList:
            if file_name.endswith(".jpg") or file_name.endswith(".jpeg") or file_name.endswith(".png"):
                # original_file_path = total_path+"/"+file_name
                original_file_path = os.path.join(total_path, file_name)
                if file_name.endswith(".jpeg"):
                    save_image_path = split_image_name+"/"+file_name.split(".jpeg")[0]+".jpg"
                    # save_image_path = os.path.join(split_image_name, file_name.split(".jpeg")[0]+".jpg"
                elif file_name.endswith(".png"):
                    save_image_path = split_image_name+"/"+file_name.split(".png")[0]+".png"
                elif file_name.endswith(".jpg"):
                    save_image_path = split_image_name + "/" + file_name.split(".jpg")[0] + ".jpg"
                # image_path = split_image_name+"/"+file_name
                image=cv2.imread(original_file_path)
                cv2.imwrite(save_image_path,image)
                # copyfile( original_file_path,split_image_name+"/"+file_name);
            elif file_name.endswith(".xml"):
                original_file_path = total_path+"/"+file_name
                copyfile(original_file_path,split_result_name+"/"+file_name);
            count += 1
            print(count)

# 根据图像删除多余的xml
def compare_image_label_remove_xml(dir):
    image_label_split_path = os.path.join(dir, 'images_label_split')
    images_label_file_List = os.listdir(image_label_split_path)
    print("filenames : {}".format(images_label_file_List))
    for filename in images_label_file_List:
        print("processing {}".format(filename))
        xml_dir = os.path.join(image_label_split_path, filename, 'Annotations')
        image_dir = os.path.join(image_label_split_path, filename, 'JPEGImages')

        count = 0
        xml_list = os.listdir(xml_dir)
        for xml_name in tqdm(xml_list):
            each_name, suffix = os.path.splitext(xml_name)
            image_name = os.path.join(image_dir, '{}.jpg'.format(each_name))
            xml_name = os.path.join(xml_dir, '{}.xml'.format(each_name))

            if os.path.exists(xml_name) and not os.path.exists(image_name):
                print("remove {}".format(xml_name))
                os.remove(xml_name)
            count += 1
        print("compare_image_label_remove_xml is over!!!")

# 根据xml删除多余的image
def compare_image_label_remove_image(dir):
    image_label_split_path = os.path.join(dir, 'images_label_split')
    images_label_file_List = os.listdir(image_label_split_path)
    print("filenames : {}".format(images_label_file_List))
    for filename in images_label_file_List:
        print("processing {}".format(filename))
        xml_dir = os.path.join(image_label_split_path, filename, 'Annotations')
        image_dir = os.path.join(image_label_split_path, filename, 'JPEGImages')
        image_List = os.listdir(image_dir)
        count = 0
        for filename in tqdm(image_List):
            each_name, suffix = os.path.splitext(filename)
            image_name = os.path.join(image_dir, '{}{}'.format(each_name, suffix))
            # xml_name = xml_dir + "/%s.xml" % (each_name)
            xml_name = os.path.join(xml_dir, '{}.xml'.format(each_name))
            if not os.path.exists(image_name) or not os.path.exists(xml_name):
                print("remove {}".format(image_name))
                os.remove(image_name)
            count += 1
        print("compare_image_label_remove_image is over!!!")

#  将所有xml汇总起来
def collect_xml(dir):
    ## 新建xml_total文件夹
    xml_total_dir = dir + "xml_total"
    isExists = os.path.exists(xml_total_dir)
    if not isExists:
        os.makedirs(xml_total_dir)
        print(str(xml_total_dir) + "新建成功")
    ## 读取分开的每个文件夹
    image_label_split_path = dir + "images_label_split"
    images_label_file_List = os.listdir(image_label_split_path)
    count = 0
    for filename in images_label_file_List:
        xml_dir = image_label_split_path + "/" + filename + "/results"
        xml_list = os.listdir(xml_dir)
        for xml_name in xml_list:
            each_name = xml_name.split(".xml")[0]
            xml_read_name = xml_dir + "/%s.xml" % (each_name)
            xml_wrtite_name = xml_total_dir + "/%s.xml" % (each_name)
            copyfile(xml_read_name, xml_wrtite_name)
            count += 1
            print("copy_xml_count:" + str(count))


# 将所有的image汇总起来
def collect_image(dir):
    ## 新建xml_total文件夹
    image_total_dir = dir + "image_total"
    isExists = os.path.exists(image_total_dir)
    if not isExists:
        os.makedirs(image_total_dir)
        print(str(image_total_dir) + "新建成功")
    ## 读取分开的每个文件夹
    image_label_split_path = dir + "images_label_split"
    images_label_file_List = os.listdir(image_label_split_path)
    count = 0
    for filename in images_label_file_List:
        image_dir = image_label_split_path + "/" + filename + "/images"
        image_list = os.listdir(image_dir)
        for image_name in image_list:
            each_name = image_name.split(".jpg")[0]
            image_read_name = image_dir + "/%s.jpg" % (each_name)
            image_wrtite_name = image_total_dir + "/%s.jpg" % (each_name)
            copyfile(image_read_name, image_wrtite_name)
            count += 1
            print("copy_image_count:" + str(count))

def remove_not_satisfied_xml(dir):
    image_label_split_path = os.path.join(dir, 'images_label_split')
    images_label_file_List = os.listdir(image_label_split_path)
    print("filenames : {}".format(images_label_file_List))
    for filename in images_label_file_List:
        print("processing {}".format(filename))
        xml_dir = os.path.join(image_label_split_path, filename, 'Annotations')

        xml_count = 0
        xml_List = os.listdir(xml_dir)
        for xml_name in tqdm(xml_List):
            if xml_name == ".idea":continue
            file_result_path = os.path.join(xml_dir ,xml_name)
            ## 进行判断
            xml_path = os.path.join(xml_dir , xml_name)
            # print(xml_path)
            tree = ET.parse(xml_path)
            if tree is None:
                print("{} is None".format(xml_path))
                os.remove(xml_path)
                # os.system('rm -rf {}'.format(path_remake(xml_path)))
            roots = tree.getroot()
            for element in roots.findall('object'):
                bndname = element.find('name').text
                bndbox_xmin = element.find('bndbox').find("xmin").text
                bndbox_ymin = element.find('bndbox').find("ymin").text
                bndbox_xmax = element.find('bndbox').find("xmax").text
                bndbox_ymax = element.find('bndbox').find("ymax").text
                # 像素尺寸大于40×40
                bndbox_width = int(bndbox_xmax) - int(bndbox_xmin)
                # bndbox_height = int(bndbox_ymax) - int(bndbox_ymin)
                if bndbox_width < 1:
                    roots.remove(element)
            xml_count += 1
            # print("v3:" + str(xml_count))
            tree.write(file_result_path)

def remove_image_null_xml(dir,label_list):
    image_label_split_path = os.path.join(dir, 'images_label_split')
    images_label_file_List = os.listdir(image_label_split_path)
    print("filenames : {}".format(images_label_file_List))
    for filename in images_label_file_List:
        print("processing {}".format(filename))
        xml_dir = os.path.join(image_label_split_path, filename, 'Annotations')
        image_dir = os.path.join(image_label_split_path, filename, 'JPEGImages')

        xml_count = 0
        xmls_List = os.listdir(xml_dir)
        for xml_name in tqdm(xmls_List):
            if xml_name == ".idea": continue
            delete_confirm = True
            ## 进行判断
            xml_path = os.path.join(xml_dir , xml_name)
            tree = ET.parse(xml_path)
            roots = tree.getroot()
            for element in roots.findall('object'):
                bndname = element.find('name').text
                if bndname in label_list:
                    delete_confirm = False
            if delete_confirm==True:
                # 删除xml_clean
                os.remove(xml_path)
                # 删除image
                image = os.path.join(image_dir , xml_name.split(".xml")[0]+".jpg")
                os.remove(image)
                # os.system('rm -rf {}'.format(image))
                print("删除"+str(xml_path)+"成功")
            xml_count += 1

def show_label(dir,label_list, num = 1000, save=False):
    image_label_split_path = os.path.join(dir, 'images_label_split')
    images_label_file_List = os.listdir(image_label_split_path)
    print("filenames : {}".format(images_label_file_List))
    for filename in images_label_file_List:
        print("processing {}".format(filename))
        xml_dir = os.path.join(image_label_split_path, filename, 'Annotations')
        image_dir = os.path.join(image_label_split_path, filename, 'JPEGImages')
        xmls_List = os.listdir(xml_dir)
        for xml_name in xmls_List:
            xml_path = os.path.join(xml_dir , xml_name)
            print("xml_path:"+str(xml_path))
            image_path = os.path.join(image_dir , xml_name.split(".xml")[0]+".jpg")
            image = cv2.imread(image_path)
            tree = ET.parse(xml_path)
            roots = tree.getroot()
            for element in roots.findall('object'):
                bndname = element.find('name').text
                if bndname in label_list:
                    bndbox_xmin = element.find('bndbox').find("xmin").text
                    bndbox_ymin = element.find('bndbox').find("ymin").text
                    bndbox_xmax = element.find('bndbox').find("xmax").text
                    bndbox_ymax = element.find('bndbox').find("ymax").text
                    for i in range(len(label_list)):
                        if bndname == label_list[i]:
                                cv2.rectangle(image, (int(bndbox_xmin), int(bndbox_ymin)), (int(bndbox_xmax), int(bndbox_ymax)),
                                      colors_list[i], 2)

            image_draw = cv2.resize(image, (1200, 900))
            cv2.imshow("image", image_draw)
            cv2.waitKey(0)

# 小类合并到大类                
# 0 : [car, moto, trunk ....]
# 1 : [person, staff ...]
def merge_classes(name, classes_id):
    for key, values in classes_id.items():
        if name in values:
            return key
        # else:
        #     return None

## yolov3的生成txt
def yolov3_get_txt(dir,label_list):
    # 判断txt的文件夹是否存在，不存在则新建
    txt_total = os.path.join(dir , "image_txt")
    isExists = os.path.exists(txt_total)
    if not isExists:
        os.makedirs(txt_total)
        print(str(txt_total) + "新建成功")

    # train中xml文件路径
    xmldir = os.path.join(dir , "train", "Annotations")
    trainList = os.listdir(xmldir)
    # xml_id = 0
    # if classes_id is not None:
    #     train_classes_id_count = {}
    #     for tkey in classes_id.keys():
    #         train_classes_id_count[tkey] = 0

    train_label_count = {}
    for lb in label_list:
        train_label_count[lb] = 0

    for filename in tqdm(trainList):

        txtname = filename.split('xml')[0] + 'txt'
        # print(filename)
        txt_file = open(os.path.join(txt_total, txtname), 'w')
        xmlname = os.path.join(xmldir, filename)
        # print(xmlname)
        tree = ET.parse(xmlname)
        roots = tree.getroot()
        image_path = os.path.join(dir , "train", "JPGImages", str(filename.split('.xml')[0])+".jpg")
        image = cv2.imread(image_path)
        height,width,_=image.shape
        for element in roots.findall('object'):
            name = element.find('name').text
            # if name == "other":
            #     continue
            if name not in label_list:
                continue
            bndbox = element.find('bndbox')
            xmin = float(bndbox.find('xmin').text)
            ymin = float(bndbox.find('ymin').text)
            xmax = float(bndbox.find('xmax').text)
            ymax = float(bndbox.find('ymax').text)
            xcenter = (xmin + xmax) / 2
            ycenter = (ymin + ymax) / 2
            w = float(xmax - xmin)
            h = float(ymax - ymin)
            xcenter_percent = xcenter / width
            ycenter_percent = ycenter / height
            w_percent = w / width
            h_percent = h / height
            # if classes_id is not None:
            #     class_id = merge_classes(name, classes_id)
            #     text_file = str(class_id) + ' ' + str(xcenter_percent) + ' ' + str(ycenter_percent) + ' ' + str(w_percent) + ' ' + str(h_percent) + '\n'
            # else:
            text_file = str(label_list.index(name)) + ' ' + str(xcenter_percent) + ' ' + str(ycenter_percent) + ' ' + str(w_percent) + ' ' + str(h_percent) + '\n'
            txt_file.write(text_file)

           
            # if name in label_list:
            #     train_label_count[name] += 1

            # if classes_id is not None:
            #     for k, s in classes_id.items():
            #         if name in s:
            #             train_classes_id_count[k] += 1
            # if name == "head_shoulder":
            #     headshouler_train += 1
            # elif name == 'guard_shoulder':
            #     guardshouler_train += 1
            # elif name == 'helmet_shoulder':
            #     helmetshouler_train += 1



        # xml_id += 1
        # print(xml_id)
        txt_file.close()

    # test中xml文件路径
    xmldir = os.path.join(dir , "test", "Annotations")
    testList = os.listdir(xmldir)
    xml_id = 0
    # test_headshoulder = 0
    # test_guardshoulder = 0
    # test_helmetshoulder = 0
    test_label_count = {}
    for lb in label_list:
        test_label_count[lb] = 0

    # if classes_id is not None:
    #     test_classes_id_count = {}
    #     for tkey in classes_id.keys():
    #         test_classes_id_count[tkey] = 0

    for filename in tqdm(testList):
        txtname = filename.split('xml')[0] + 'txt'
        txt_file = open(os.path.join(txt_total, txtname), 'w')
        xmlname = os.path.join(xmldir, filename)
        # print(xmlname)
        tree = ET.parse(xmlname)
        roots = tree.getroot()
        image_path = os.path.join(dir , "test","JPGImages" , str(filename.split('.xml')[0]) + ".jpg")
        image = cv2.imread(image_path)
        if image is None:
            print("NoneTyep : {}".format(image_path))

        height, width, _ = image.shape
        for element in roots.findall('object'):
            name = element.find('name').text
            # if name == "other":
            #     continue
            if name not in label_list:
                continue
            bndbox = element.find('bndbox')
            xmin = float(bndbox.find('xmin').text)
            ymin = float(bndbox.find('ymin').text)
            xmax = float(bndbox.find('xmax').text)
            ymax = float(bndbox.find('ymax').text)
            xcenter = (xmin + xmax) / 2
            ycenter = (ymin + ymax) / 2
            w = float(xmax - xmin)
            h = float(ymax - ymin)
            xcenter_percent = xcenter / width
            ycenter_percent = ycenter / height
            w_percent = w / width
            h_percent = h / height
            # if classes_id is not None:
            #     class_id = merge_classes(name, classes_id)
            #     text_file = str(class_id) + ' ' + str(xcenter_percent) + ' ' + str(ycenter_percent) + ' ' + str(w_percent) + ' ' + str(h_percent) + '\n'
            # else:
            text_file = str(label_list.index(name)) + ' ' + str(xcenter_percent) + ' ' + str(ycenter_percent) + ' ' + str(w_percent) + ' ' + str(h_percent) + '\n'
            txt_file.write(text_file)

            if name in label_list:
                test_label_count[name] += 1
            #
            # if classes_id is not None:
            #     for k, s in classes_id.items():
            #         if name in s:
            #             test_classes_id_count[k] += 1

            # if name == 'head_shoulder':
            #     test_headshoulder += 1
            # elif name == 'guard_shoulder':
            #     test_guardshoulder += 1
            # elif name == 'helmet_shoulder':
            #     test_helmetshoulder += 1
            # # smoke_count_test += 1
        xml_id += 1
        # print(xml_id)
        txt_file.close()
    # print_info = 'train examples'
    # for key, value in train_label_count.items():
    #     print_info = print_info + "\n\t{}:\t{}".format(key, value)
    # print_info = print_info + '\n'
    #
    # print(print_info)

    # if classes_id is not None:
    #     print_info = 'train merge examples'
    #     for key, value in train_classes_id_count.items():
    #         print_info = print_info + "\n\t{}:\t{}".format(key, value)
    #     print_info = print_info + '\n'
    #     print(print_info)
    #
    # print_info = 'test examples'
    # for key, value in test_label_count.items():
    #     print_info = print_info + "\n\t{}:\t{}".format(key, value)
    # print_info = print_info + '\n'
    #
    # print(print_info)
    #
    # if classes_id is not None:
    #     print_info = 'test merge examples'
    #     for key, value in test_classes_id_count.items():
    #         print_info = print_info + "\n\t{}:\t{}".format(key, value)
    #     print_info = print_info + '\n'
    #     print(print_info)
    #
    # # print('train examples:\n\t{}:{}\n\t{}:{}\n\t{}:{}\n'.format('head_shoulder', headshouler_train, 'guard_shoulder', guardshouler_train, 'helmet_shoulder', helmetshouler_train))
    #
    # # print('test examples:\n\t{}:{}\n\t{}:{}\n\t{}:{}\n'.format('head_shoulder', test_headshoulder, 'guard_shoulder', test_guardshoulder, 'helmet_shoulder', test_helmetshoulder))


def yolov3_get_train_test_txt(dir):
    train_xml_path = os.path.join(dir,"train","Annotations")
    test_xml_path = os.path.join(dir , "test","Annotations")

    # 需要填写的路径
    train_txt_image_path = os.path.join(dir , "image_txt")

    # 将train和test需要保存的路径
    train_test_txt_path = os.path.join(dir , "train_test_txt")
    isExists = os.path.exists(train_test_txt_path)
    if not isExists:
        os.makedirs(train_test_txt_path)
        print(str(train_test_txt_path) + "新建成功")

    # 将train的生成txt
    xml_List = os.listdir(train_xml_path)
    train_file = open(os.path.join(train_test_txt_path, 'train.txt'), 'w')
    count =0
    for filename in xml_List:
        file_name = filename.split(".xml")[0]
        train_write_name =  os.path.join(train_txt_image_path, file_name+".jpg")+ '\n'
        train_file.write(train_write_name)
        count+=1
        print("train:"+str(count))
    train_file.close()

    # 将test的生成txt
    xml_List = os.listdir(test_xml_path)
    test_file = open(os.path.join(train_test_txt_path, 'test.txt'), 'w')
    count = 0
    for filename in xml_List:
        file_name = filename.split(".xml")[0]
        test_write_name = os.path.join(train_txt_image_path , file_name + ".jpg") + '\n'
        test_file.write(test_write_name)
        count += 1
        print("test:" + str(count))
    test_file.close()





    # txt_image_total
    # txt_image_total_path = dir+"image_txt"
    # isExists = os.path.exists(txt_image_total_path)
    # if not isExists:
    #     os.makedirs(txt_image_total_path)
    #     print(str(txt_image_total_path) + "新建成功")
    # # train_test_txt_path
    # train_test_txt_path = dir+"train_test_txt"
    # isExists = os.path.exists(train_test_txt_path)
    # if not isExists:
    #     os.makedirs(train_test_txt_path)
    #     print(str(train_test_txt_path) + "新建成功")
    # # image_total
    # image_total_path = dir+"image_total"
    #
    # # 打开train.txt和test.txt进行写入
    # test_file = open(os.path.join(train_test_txt_path, 'test.txt'), 'w')
    # train_file = open(os.path.join(train_test_txt_path, 'train.txt'), 'w')
    # img_list = os.listdir(image_total_path)
    # random.shuffle(img_list)
    # num = 0
    # for imgname in img_list:
    #     if num < test_count:
    #         text = txt_image_total_path+"/" + imgname + '\n'
    #         test_file.write(text)
    #     else:
    #         text = txt_image_total_path+"/" + imgname + '\n'
    #         train_file.write(text)
    #     num += 1
    #     print(num)
    # test_file.close()
    # train_file.close()

import shutil
# 将image和txt移动到文件夹中
def yolov3_move_image(dir):

    train_images_path = os.path.join(dir, "train", 'JPGImages')
    
    test_images_path = os.path.join(dir , "test","JPGImages")

    train_count = 0
    train_images_list = os.listdir(train_images_path)
    for train_image_name in train_images_list:
        train_image_path = os.path.join(train_images_path , train_image_name)
        train_image_save_path = os.path.join(dir , "image_txt" , train_image_name)
        if os.path.exists(train_image_save_path):
            continue
        else:
            shutil.copy(train_image_path,train_image_save_path)
            train_count+=1
            print("tain_count:"+str(train_count))

    test_count = 0
    test_images_list = os.listdir(test_images_path)
    for test_image_name in test_images_list:
        test_image_path = os.path.join(test_images_path , test_image_name)
        test_image_save_path = os.path.join(dir , "image_txt" , test_image_name)
        if os.path.exists(test_image_save_path):
            continue
        else:
            shutil.copy(test_image_path, test_image_save_path)
            test_count += 1
            print("test_count"+ str(test_count))


def yolov3_get_train_test_file(dir,val_percent):
    # 新建train的文件夹
    train_file_path = os.path.join(dir , "train")
    train_JPGImages = os.path.join(train_file_path , "JPGImages")
    train_Annotations = os.path.join(train_file_path , "Annotations")

    # For Train
    isExists = os.path.exists(train_file_path)
    if not isExists:
        os.makedirs(train_file_path)
        os.makedirs(train_JPGImages)
        os.makedirs(train_Annotations)
        print(str(train_file_path) + "新建成功")
    # 新建test的文件夹
    test_file_path = os.path.join(dir , "test")
    test_JPGImages = os.path.join(test_file_path , "JPGImages")
    test_Annotations = os.path.join(test_file_path , "Annotations")
    isExists = os.path.exists(test_file_path)
    # For Test
    if not isExists:
        os.makedirs(test_file_path)
        os.makedirs(test_JPGImages)
        os.makedirs(test_Annotations)
        print(str(test_file_path) + "新建成功")

    # 读取xml路径
    image_label_split_path = os.path.join(dir , "images_label_split")
    images_label_file_List = os.listdir(image_label_split_path)
    count = 0

    print("filenames : {}".format(images_label_file_List))
    for filename in images_label_file_List:
        print('processing {}'.format(filename))
        xml_dir = os.path.join(image_label_split_path , filename , "Annotations")

        xmls_List = os.listdir(xml_dir)
        print("len xmls_list {}".format(len(xmls_List)))

        # 后续需要改进
        # 检查数据集是否在train或者test存在，如果存在则跳过不加入到train或test集中
        sel_xmls_List = []
        for xml_name in xmls_List:
            train_xml_path = os.path.join(train_Annotations , filename+'_' + xml_name)
            test_xml_path = os.path.join(test_Annotations , filename+'_' + xml_name)

            if os.path.exists(train_xml_path) or os.path.exists(test_xml_path):
                continue
            else:
                sel_xmls_List.append(xml_name)

        xmls_List = sel_xmls_List
        print("len xmls_list {}".format(len(xmls_List)))

        random.shuffle(xmls_List)

        val_number = int(val_percent*len(xmls_List))
        for i in tqdm(range(len(xmls_List))):
            if xmls_List[i]==".idea":continue
            xml_path = os.path.join(xml_dir , xmls_List[i])
            if i <= val_number:
                xml_save_path = os.path.join(test_Annotations , filename+'_'+xmls_List[i])
            else:
                xml_save_path = os.path.join(train_Annotations , filename+'_'+xmls_List[i])
            shutil.copy(xml_path, xml_save_path)

            image_path = os.path.join(image_label_split_path , filename ,  "JPEGImages" , xmls_List[i].split(".xml")[0] + ".jpg")
            if i <= val_number:
                image_save_path = os.path.join(test_JPGImages , filename+'_'+xmls_List[i].split(".xml")[0] + ".jpg")
            else:
                image_save_path = os.path.join(train_JPGImages , filename+'_'+xmls_List[i].split(".xml")[0] + ".jpg")
            shutil.copy(image_path, image_save_path)

            count+=1
    print(count)


def yolov3_test_add_image_xml(dir,add_path):
    add_img_list = os.listdir(add_path)
    if len(add_img_list)>0:
        for filename in add_img_list:
            file_path = add_path+filename
            # 将image复制到test/JPGimages和image_total中
            if file_path[-4::]==".jpg":
                test_image_path = dir + "test/JPGImages/" + filename.split(".jpg")[0] + ".jpg"
                image_total_path = dir + "image_total/" + filename.split(".jpg")[0] + ".jpg"
                copyfile(file_path,test_image_path)
                copyfile(file_path, image_total_path)
            # 将xml复制到test/Annoations中
            if file_path[-4::]==".xml":
                test_xml_path = dir + "test/Annotations/" + filename.split(".xml")[0] + ".xml"
                copyfile(file_path, test_xml_path)
            print(str(file_path))



# edit by my
# 目标检测源文件文件处理脚本
# 遍历文件夹
def walkFile(file):
    all_list = []
    for root, dirs, files in os.walk(file):

    # root 表示当前正在访问的文件夹路径
    # dirs 表示该文件夹下的子目录名list
    # files 表示该文件夹下的文件list

        # 遍历文件
        for f in files:
            all_list.append(os.path.join(root, f))

        # 遍历所有的文件夹
        for d in dirs:
            all_list.append(os.path.join(root, d))
    return all_list

# 计算样本数
def count_labels_to_csv(src_dir, det_dir):
    """
    src_dir : 统计xml文件源目录
    det_dir : 保存统计结果为csv格式路径
    遍历src_dir文件下所有xml文件，将xml文件生成csv文件并统计目标检测类别的数量。
    支持两种文件结构类型：
    1 src_dir
        |-- file_name1
        |       |-- images
        |       |-- results
        |-- file_name2
        ...

    2 src_dir
        |-- Annotations
        |-- JPGImages
    """
    xmls_list = walkFile(src_dir)
    # xmls_list = []
    # imgs_list = []

    xmls_list = [l for l in xmls_list if os.path.splitext(l)[-1].lower() in xml_format]
    
    label_count = {}
    f = open(det_dir, 'w')
    f.writelines("path,img_h,img_w,obj_name,xmin,ymin,xmax,ymax\n")
    for xml_path in tqdm(xmls_list):
        xmls_folder_name, xml_name = os.path.split(xml_path)
        file_dir, xmls_file_name = os.path.split(xmls_folder_name)
        
        img_name = os.path.splitext(xml_name)[0] + '.jpg'
        #常规下 images和results在同一个目录下；JPGImages和Annotations在同一目录下
        if xmls_file_name == 'results':
            img_path = os.path.join(file_dir, 'images', img_name)
        elif xmls_file_name == 'Annotations':
            img_path = os.path.join(file_dir, 'JPGImages', img_name)
        
        image = cv2.imread(img_path)
        h,w,c = image.shape
        tree = ET.parse(xml_path)
        roots = tree.getroot()
        for element in roots.findall('object'):
            bndname = element.find('name').text
            if bndname not in label_count.keys():
                label_count[bndname] = 1
            else:
                label_count[bndname] += 1

            bndbox = element.find('bndbox')
            xmin = int(bndbox.find('xmin').text)
            ymin = int(bndbox.find('ymin').text)
            xmax = int(bndbox.find('xmax').text)
            ymax = int(bndbox.find('ymax').text)
            # coord = (xmin,ymin,xmax,ymax)
            f.writelines("{},{},{},{},{},{},{},{}\n".format(xml_path, h,w,bndname, xmin, ymin, xmax, ymax))

    print_info = 'check and count dir :{}'.format(src_dir)
    for key, value in label_count.items():
        print_info = print_info + "\n\t{}:\t{}".format(key, value)    
    print_info = print_info + '\n'
    print(print_info)


if __name__ == '__main__':
    src_dir = '/mnt/mnt190/mayue/Datasets/gas_station/test'
    det_dir = '/mnt/mnt190/mayue/Datasets/gas_station/test.csv'
    count_labels_to_csv(src_dir, det_dir)
    

