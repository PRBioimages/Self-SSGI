import glob
import os

import timm
import cv2
from torch.utils.data import IterableDataset, DataLoader
import sio as sio
import torch.nn.functional as F
import numpy as np
import torch
import random
from scipy.io import savemat
import scipy.io as sio
from torch import nn
from torch.nn.parameter import Parameter



class ExtendedCELLResNet50(nn.Module):
    def __init__(self, base_model, mlp_hidden_dim=128, out_dim=10):
        super().__init__()
        self.base_model = base_model
        self.mlp = nn.Sequential(
            nn.Linear(2048, mlp_hidden_dim),
            #nn.ReLU(),
            #nn.Linear(mlp_hidden_dim, out_dim)
        )

    def forward(self, x):
        output, pooled_features = self.base_model(x)
        mlp_output = self.mlp(pooled_features)
        return mlp_output

def gem(x, p=3, eps=1e-6):
    return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(1. / p)


class GeM(nn.Module):
    def __init__(self, p=3, eps=1e-6):
        super(GeM, self).__init__()
        self.p = Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        return gem(x, p=self.p, eps=self.eps)

    def __repr__(self):
        return self.__class__.__name__ + '(' + 'p=' + '{:.4f}'.format(self.p.data.tolist()[0]) + ', ' + 'eps=' + str(
            self.eps) + ')'


class CELLResNet50(nn.Module):
    def __init__(self, model_name='resnet200d', out_dim=19, pretrained=False, dropout=0.5,
                 pool='AdaptiveAvgPool2d'):
        super().__init__()
        self.model = timm.create_model(model_name.split('_')[-1], pretrained=pretrained, in_chans=4)
        n_features = self.model.fc.in_features
        self.model.global_pool = nn.Identity()
        self.model.fc = nn.Identity()
        if pool == 'AdaptiveAvgPool2d':
            self.pooling = nn.AdaptiveAvgPool2d(1)
        elif pool == 'gem':
            self.pooling = GeM()
        self.fc = nn.Linear(n_features, out_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        bs = x.size(0)
        features = self.model(x)
        pooled_features = self.pooling(features).view(bs, -1)
        output = self.fc(self.dropout(pooled_features))
        return output,pooled_features

    @property
    def net(self):
        return self.model

def main(img_dir,save_dir):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    all_file_list = glob.glob(os.path.join(img_dir, "*.jpg"))

    model_path='pretrained model path'
    base_model = CELLResNet50(pretrained=False, model_name='Cell_resnet50d')
    model_test = ExtendedCELLResNet50(base_model, mlp_hidden_dim=128, out_dim=10).to(device)
    checkpoint = torch.load(model_path)
    model_test.load_state_dict(checkpoint['net'])


    model_test.eval()

    with torch.no_grad():
        for file_path in all_file_list:
            current_img = sio.loadmat(file_path)
            current_img = current_img['img']
            current_img = current_img / 255.0
            for j in range(4):  # Scaling for each channel
                current_img[ :, :, j] = cv2.resize(current_img[:, :, j], (224, 224))
                images_encoding = model_test(current_img.to(device))

            images_encoding = np.asarray(images_encoding.cpu().detach(), dtype=np.float32)

            file_name = os.path.basename(file_path)
            file_name_no_ext = os.path.splitext(file_name)[0]
            file_name = save_dir+file_name_no_ext[0]+".mat"
            savemat(file_name, {'images_encoding': images_encoding})






if __name__=='__main__':
    img_dir='your images folder'
    save_dir='your save images features folder'
    main(img_dir,save_dir)






