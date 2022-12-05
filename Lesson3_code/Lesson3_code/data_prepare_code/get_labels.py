import xml.etree.ElementTree as ET
import argparse
import os
from tqdm import tqdm

def get_label_set(opt):
    all_labels_dict = {}
    xmls_file_list = os.listdir(opt.dir)

    for xml_file_name in xmls_file_list:
        xml_file_dir = os.path.join(opt.dir ,xml_file_name, 'Annotations')
        xmls_list = os.listdir(xml_file_dir)
        for xml_name in tqdm(xmls_list):
            base_name, suffix = os.path.splitext(xml_name)
            xml_path = os.path.join(xml_file_dir, xml_name)
            if suffix != '.xml':
                print('ilegal xml file : {}'.format(xml_name))
                continue
            tree = ET.parse(xml_path)
            roots = tree.getroot()
            for element in roots.findall('object'):
                bndname = element.find('name').text
                if not all_labels_dict.get(bndname):
                    all_labels_dict[bndname] = 1
                else:
                    all_labels_dict[bndname] += 1
        total = 0
        for key, value in all_labels_dict.items():
            print("\n{} : {}".format(key, value))
            total += value
        print('total examples : {}'.format(total))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', type=str, default='C:/Users/86151/Desktop/aidlux/lesson/Lesson3/Yolov5_train/data/head', help='data dir')
    parser.add_argument('--get_labels', type=bool, default=True, help='get dataset data labels')
    opt = parser.parse_args()
    get_label_set(opt)
