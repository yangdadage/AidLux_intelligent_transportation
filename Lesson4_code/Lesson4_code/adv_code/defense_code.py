import os
import torch
import torch.nn as nn
from torchvision.models import mobilenet_v2
from advertorch.utils import predict_from_logits
from advertorch.utils import NormalizeByChannelMeanStd
from robust_layer import GradientConcealment, ResizedPaddingLayer

from advertorch.attacks import LinfPGDAttack
from advertorch_examples.utils import ImageNetClassNameLookup
from advertorch_examples.utils import bhwc2bchw
from advertorch_examples.utils import bchw2bhwc

device = "cuda" if torch.cuda.is_available() else "cpu"


### 读取图片
def get_image():
    img_path = os.path.join(
        "D:/python_project/jiang_dabai/intelligent_transportation/Lesson4_code/Lesson4_code/adv_code/images",
        "cab.png")
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

### GCM模块
robust_mode = GradientConcealment()


### 常规模型+GCM模块
class Model(nn.Module):
    def __init__(self, l=290):
        super(Model, self).__init__()

        self.l = l
        self.gcm = GradientConcealment()
        # model = resnet18(pretrained=True)
        model = mobilenet_v2(weights=None)
        model_param = torch.load(
            f'D:/python_project/jiang_dabai/intelligent_transportation/Lesson4_code/'
            f'Lesson4_code/model/mobilenet_v2-b0353104.pth'
        )
        model.load_state_dict(model_param)

        # pth_path = "/Users/rocky/Desktop/训练营/model/mobilenet_v2-b0353104.pth"
        # print(f'Loading pth from {pth_path}')
        # state_dict = torch.load(pth_path, map_location='cpu')
        # is_strict = False
        # if 'model' in state_dict.keys():
        #    model.load_state_dict(state_dict['model'], strict=is_strict)
        # else:
        #    model.load_state_dict(state_dict, strict=is_strict)

        normalize = NormalizeByChannelMeanStd(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.model = nn.Sequential(normalize, model)

    def load_params(self):
        pass

    def forward(self, x):
        x = self.gcm(x)
        # x = ResizedPaddingLayer(self.l)(x)
        out = self.model(x)
        return out


### 常规模型+GCM模块 加载
model_defense = Model().eval().to(device)

### 数据预处理
np_img = get_image()
img = torch.tensor(bhwc2bchw(np_img))[None, :, :, :].float().to(device)
imagenet_label2classname = ImageNetClassNameLookup()

### 测试模型输出结果
pred_defense = imagenet_label2classname(predict_from_logits(model_defense(img)))
print("test output:", pred_defense)

pre_label = predict_from_logits(model_defense(img))

### 对抗攻击：PGD攻击算法
adversary = LinfPGDAttack(
    model_defense, eps=8 / 255, eps_iter=2 / 255, nb_iter=80,
    rand_init=True, targeted=False)

### 完成攻击，输出对抗样本
advimg = adversary.perturb(img, pre_label)

### 展示源图片，对抗扰动，对抗样本以及模型的输出结果
show_images(model_defense, img, advimg)
