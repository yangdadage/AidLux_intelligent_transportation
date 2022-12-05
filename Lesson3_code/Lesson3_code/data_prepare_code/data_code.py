from xml.dom import minidom
import cv2
import os
import json
from PIL import Image
 
roadlabels = "D:/aidlux/lesson/Lesson3/lesson3_data/Crowdhuman_data/Annotations/"
roadimages = "D:/aidlux/lesson/Lesson3/lesson3_data/Crowdhuman_data/JPEGImages/"
fpath = "D:/aidlux/lesson/Lesson3/lesson3_data/Crowdhuman_data/annotation_val.odgt"
 
 
def load_func(fpath):
    assert os.path.exists(fpath)
    with open(fpath, 'r') as fid:
        lines = fid.readlines()
    records = [json.loads(line.strip('\n')) for line in lines]
    return records
 
 
bbox = load_func(fpath)
 
if not os.path.exists(roadlabels):
    os.makedirs(roadlabels)
 
for i0, item0 in enumerate(bbox):
    print(i0)
    # 建立i0的xml tree
    ID = item0['ID']  # 得到当前图片的名字
    imagename = roadimages + '/' + ID + '.jpg'  # 当前图片的完整路径，双斜杠一定要加
    savexml = roadlabels + '/' + ID + '.xml'  # 生成的.xml注释的名字，双斜杠一定要加
 
    gtboxes = item0['gtboxes']
    img_name = ID
    floder = 'CrowdHuman'
    im = cv2.imread(imagename)
    w = im.shape[1]
    h = im.shape[0]
    d = im.shape[2]
 
    doc = minidom.Document()  # 创建DOM树对象
    annotation = doc.createElement('annotation')  # 创建子节点
    doc.appendChild(annotation)  # annotation作为doc树的子节点
 
    folder = doc.createElement('folder')
    folder.appendChild(doc.createTextNode(floder))  # 文本节点作为floder的子节点
    annotation.appendChild(folder)  # folder作为annotation的子节点
 
    filename = doc.createElement('filename')
    filename.appendChild(doc.createTextNode(img_name + '.jpg'))
    annotation.appendChild(filename)
 
    source = doc.createElement('source')
    database = doc.createElement('database')
    database.appendChild(doc.createTextNode("Unknown"))
    source.appendChild(database)
    annotation.appendChild(source)
 
    size = doc.createElement('size')
    width = doc.createElement('width')
    width.appendChild(doc.createTextNode("%d" % w))
    size.appendChild(width)
    height = doc.createElement('height')
    height.appendChild(doc.createTextNode("%d" % h))
    size.appendChild(height)
    depth = doc.createElement('depth')
    depth.appendChild(doc.createTextNode("%d" % d))
    size.appendChild(depth)
    annotation.appendChild(size)
 
    segmented = doc.createElement('segmented')
    segmented.appendChild(doc.createTextNode("0"))
    annotation.appendChild(segmented)
 
    # 下面是从odgt中提取三种类型的框并转为voc格式的xml的代码
    for i1, item1 in enumerate(gtboxes):
        # 提取全身框(full box)的标注
        boxs = [int(a) for a in item1['fbox']]
        # 左上点长宽->左上右下
        minx = str(boxs[0])
        miny = str(boxs[1])
        maxx = str(boxs[2] + boxs[0])
        maxy = str(boxs[3] + boxs[1])
        # print(box)
        object = doc.createElement('object')
        nm = doc.createElement('name')
        nm.appendChild(doc.createTextNode('person'))  # 类名: fbox
        object.appendChild(nm)
        pose = doc.createElement('pose')
        pose.appendChild(doc.createTextNode("Unspecified"))
        object.appendChild(pose)
        truncated = doc.createElement('truncated')
        truncated.appendChild(doc.createTextNode("1"))
        object.appendChild(truncated)
        difficult = doc.createElement('difficult')
        difficult.appendChild(doc.createTextNode("0"))
        object.appendChild(difficult)
        bndbox = doc.createElement('bndbox')
        xmin = doc.createElement('xmin')
        xmin.appendChild(doc.createTextNode(minx))
        bndbox.appendChild(xmin)
        ymin = doc.createElement('ymin')
        ymin.appendChild(doc.createTextNode(miny))
        bndbox.appendChild(ymin)
        xmax = doc.createElement('xmax')
        xmax.appendChild(doc.createTextNode(maxx))
        bndbox.appendChild(xmax)
        ymax = doc.createElement('ymax')
        ymax.appendChild(doc.createTextNode(maxy))
        bndbox.appendChild(ymax)
        object.appendChild(bndbox)
        annotation.appendChild(object)
        savefile = open(savexml, 'w')
        savefile.write(doc.toprettyxml())
        savefile.close()
