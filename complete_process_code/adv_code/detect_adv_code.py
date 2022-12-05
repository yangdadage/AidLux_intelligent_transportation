import os
import torch
import requests
import time
import torch.nn as nn
from torchvision.models import mobilenet_v2,resnet18
from advertorch.utils import predict_from_logits
from advertorch.utils import NormalizeByChannelMeanStd
from robust_layer import GradientConcealment, ResizedPaddingLayer
from timm.models import create_model

from advertorch.attacks import LinfPGDAttack
from advertorch_examples.utils import ImageNetClassNameLookup
from advertorch_examples.utils import bhwc2bchw
from advertorch_examples.utils import bchw2bhwc

device = "cuda" if torch.cuda.is_available() else "cpu"


### 读取图片
def get_image(img_path):
    
    img_url = "https://farm1.static.flickr.com/230/524562325_fb0a11d1e1.jpg"

    def _load_image():
        from skimage.io import imread
        return imread(img_path) / 255.

    if os.path.exists(img_path):
        return _load_image()
    else:
        print(f'{img_path}不存在...')
        exit()
        import urllib
        urllib.request.urlretrieve(img_url, img_path)
        return _load_image()


def tensor2npimg(tensor):
    return bchw2bhwc(tensor[0].cpu().numpy())


normalize = NormalizeByChannelMeanStd(
    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])


### 常规模型加载
class Model(nn.Module):
    def __init__(self, l=290):
        super(Model, self).__init__()

        self.l = l
        self.gcm = GradientConcealment()
        #model = resnet18(pretrained=True)
        model = mobilenet_v2(pretrained=True)

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
        #x = self.gcm(x)
        #x = ResizedPaddingLayer(self.l)(x)
        out = self.model(x)
        return out


### 对抗攻击监测模型
class Detect_Model(nn.Module):
    def __init__(self, num_classes=2):
        super(Detect_Model, self).__init__()
        self.num_classes = num_classes
        #model = create_model('mobilenetv3_large_075', pretrained=False, num_classes=num_classes)
        model = create_model('resnet50', pretrained=False, num_classes=num_classes)

        # self.multi_PreProcess = multi_PreProcess()
        pth_path = os.path.join("/home/intelligent_transportation/Lesson5_code/model", 'track2_resnet50_ANT_best_albation1_64_checkpoint.pth')
        #pth_path = os.path.join("/Users/rocky/Desktop/训练营/Lesson5_code/model/", "track2_tf_mobilenetv3_large_075_64_checkpoint.pth")
        state_dict = torch.load(pth_path, map_location='cpu')
        is_strict = False
        if 'model' in state_dict.keys():
            model.load_state_dict(state_dict['model'], strict=is_strict)
        else:
            model.load_state_dict(state_dict, strict=is_strict)
        normalize = NormalizeByChannelMeanStd(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        # self.model = nn.Sequential(normalize, self.multi_PreProcess, model)
        self.model = nn.Sequential(normalize, model)

    def load_params(self):
        pass

    def forward(self, x):
        # x = x[:,:,32:193,32:193]
        # x = F.interpolate(x, size=(224,224), mode="bilinear", align_corners=True)
        # x = self.multi_PreProcess.forward(x)
        out = self.model(x)
        if self.num_classes == 2:
            out = out.softmax(1)
            #return out[:,1:]
            return out[:,1:]


model = Model().eval().to(device)

detect_model = Detect_Model().eval().to(device)


if __name__ == '__main__':
    from pathlib import Path
    src_img_path = Path('/home/intelligent_transportation/Lesson5_code/adv_code/orig_images/')
    # img_path_lst = [p for p in src_img_path.rglob('*') if p.name.split('_')[0] == 'adv']
    img_path_lst = [p for p in src_img_path.rglob('*') if p.name.endswith('jpg')]
    for img_path in img_path_lst:
        print('=' * 60)
        print(img_path.name) 
        np_img = get_image(img_path)
        img = torch.tensor(bhwc2bchw(np_img))[None, :, :, :].float().to(device)
        imagenet_label2classname = ImageNetClassNameLookup()
        ### 对抗攻击监测
        detect_pred = detect_model(img)
        print(detect_pred)

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
            print('=' * 60)
            print('result: ', pred)
