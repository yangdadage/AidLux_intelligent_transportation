import os
import torch
import torch.nn as nn
from torchvision.models import mobilenet_v2
from advertorch.utils import predict_from_logits
from advertorch.utils import NormalizeByChannelMeanStd

from advertorch.attacks import LinfPGDAttack
from advertorch_examples.utils import ImageNetClassNameLookup
from advertorch_examples.utils import bhwc2bchw
from advertorch_examples.utils import bchw2bhwc

device = "cuda" if torch.cuda.is_available() else "cpu"


### 读取图片
def get_image():
    img_path = os.path.join(
        "D:/python_project/jiang_dabai/intelligent_transportation/Lesson4_code/Lesson4_code/adv_code/images",
        "school_bus.png")
    img_url = "https://farm1.static.flickr.com/230/524562325_fb0a11d1e1.jpg"

    def _load_image():
        from skimage.io import imread
        return imread(img_path) / 255.

    if os.path.exists(img_path):
        return _load_image()
    else:
        import urllib
        urllib.request.urlretrieve(img_url, img_path)
        return _load_image()


def tensor2npimg(tensor):
    return bchw2bhwc(tensor[0].cpu().numpy())


### 展示攻击结果
def show_images(model, img, advimg, enhance=127):
    np_advimg = tensor2npimg(advimg)
    np_perturb = tensor2npimg(advimg - img)

    pred = imagenet_label2classname(predict_from_logits(model(img)))
    advpred = imagenet_label2classname(predict_from_logits(model(advimg)))

    import matplotlib.pyplot as plt

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


normalize = NormalizeByChannelMeanStd(
    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

### 常规模型加载
model = mobilenet_v2(weights=None)
model_param = torch.load(
    'D:/python_project/jiang_dabai/intelligent_transportation/Lesson4_code/Lesson4_code/model/mobilenet_v2-b0353104.pth'
)
model.load_state_dict(model_param)
model.eval()
model = nn.Sequential(normalize, model)
model = model.to(device)

### 数据预处理
np_img = get_image()
img = torch.tensor(bhwc2bchw(np_img))[None, :, :, :].float().to(device)
imagenet_label2classname = ImageNetClassNameLookup()

### 测试模型输出结果
pred = imagenet_label2classname(predict_from_logits(model(img)))
print("test output:", pred)

### 输出原label
pred_label = predict_from_logits(model(img))

### 对抗攻击：PGD攻击算法
adversary = LinfPGDAttack(
    model, eps=8 / 255, eps_iter=2 / 255, nb_iter=80,
    rand_init=True)

### 完成攻击，输出对抗样本
advimg = adversary.perturb(img, pred_label)

### 展示源图片，对抗扰动，对抗样本以及模型的输出结果
show_images(model, img, advimg)
