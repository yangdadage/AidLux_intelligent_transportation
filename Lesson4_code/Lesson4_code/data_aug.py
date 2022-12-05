# -*- coding: utf-8 -*-
# 杨大大哥
# 学习时间：2022年11月29日2:00
import cv2
import imageio
import numpy as np
import imgaug as ia
from imgaug import augmenters as iaa
import matplotlib.pyplot as plt


def figure_show(img):
    fig, ax = plt.subplots(figsize=(6, 6), dpi=1000)
    fig.canvas.manager.set_window_title("imgaug.imshow(%s)" % (img.shape,))
    # cmap=gray is automatically only activate for grayscale images
    ax.imshow(img, cmap="gray")
    plt.axis('off')
    plt.show()


# 设置数据增强模式
seq = iaa.Sequential([
    # iaa.Crop(px=(0, 16)),   # crop images from each side by 0 to 16px (randomly chosen)
    # iaa.Fliplr(0.5),  # horizontally flip 50% of the images
    # iaa.GaussianBlur(sigma=(0, 3.)),  # blur images with a sigma of 0 to 3.0
    # iaa.Affine(rotate=(-25, 25)),  # 设定角度
    # iaa.WithColorspace(to_colorspace="HSV"),  # HSV色彩空间
    # iaa.AddToHueAndSaturation((-20, 20), per_channel=True),  # 随机色调和饱和度

    # crop and pad images 裁剪并填充图形
    iaa.CropAndPad(percent=(-0.2, 0.2), pad_mode="edge"),
    # change their color 更改颜色
    # iaa.AddToHueAndSaturation((-60, 60)),
    # water-like effect 添加水体般的效果
    iaa.ElasticTransformation(alpha=90, sigma=9),
    # replace one squared area within the image by a constant intensity value 用一个单色框随机填充区域
    iaa.Cutout()



], random_order=True  # 随机排序使用各种增强技术
)


image = imageio.v2.imread('../../Lesson3_code/Lesson3_code/src_img.png')
images_aug = seq.augment_image(image)
# image = cv2.cvtColor(images_aug, cv2.COLOR_BGR2RGB)

# cv2.imwrite('test.jpg', image)
figure_show(image)
figure_show(images_aug)


exit()
# 增强多张图片
image = imageio.v2.imread('../../Lesson3_code/Lesson3_code/src_img.png')
images = [image, image, image, image]
images_aug = seq.augment_images(images)
# image = cv2.cvtColor(images_aug, cv2.COLOR_BGR2RGB)

# cv2.imwrite('test.jpg', image)
ia.imshow(image, )
ia.imshow(np.hstack(images_aug))

