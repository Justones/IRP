import torch
import os
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from data import DataSet, pil_loader
#from unet import UNet
from model import *
from glob import glob
import torchvision.transforms as TF
from utils import save_image, load_checkpoint, load_image, make_path, token_cmp, token_cmp1
from scipy import stats

config = {
    'data_path': './',
    'save_path': './results/',
    'lr': 5e-5,
    'epoch': 30,
    'batch_size': 16,
    'patch_size': 256,
    'ckpt': './ckpt/',
    'ratio': 'test2',
    'result_path': './results/',
    'save_frequency': 10,
    'Resume': False,
    'weights_path': './ckpt/test2/epoch_30_join.pt',
}

test_dataset = DataSet(phase='test', path = config['data_path'])

test_data_loader = DataLoader(test_dataset, num_workers=8, batch_size=1, shuffle = True)

if not os.path.exists(config['save_path']+ str(config['ratio'])):
    os.makedirs(config['save_path']+ str(config['ratio']))

model = IPR()
model = model.cuda()

load_checkpoint(model,checkpoint_path=config['weights_path'])
model.eval()
dir_list = glob(config['data_path'] + '*/')

# transforms = TF.Compose([
#                     TF.ToTensor(),
#                     TF.Normalize(mean=(0.485, 0.456, 0.406),
#                                 std=(0.229, 0.224, 0.225))
#                 ])

all_list = []


k = 1
b = 0

# with torch.no_grad():

for inp_img, inp_img_exp, label, path in test_data_loader:
    inp_img, inp_img_exp, label = inp_img.cuda(), inp_img_exp.cuda(), label.cuda()
    f_q, f_x, q = model(inp_img, inp_img_exp)

    for idx in range(len(label)):
        all_list.append([path[idx], float(q[idx]), float(label[idx])])

print(all_list)
all_list.sort(key=token_cmp)
sum_srcc = 0
sum_plcc = 0
sum_flag = 0
x = []
y = []
count = 0

for idx in range(0, len(all_list), 11):
    temp_list = all_list[idx:idx + 11]
    temp_list.sort(key=token_cmp1)
    out_coff = []
    lab_coff = []
    # print(temp_list[0][0])
    for a in temp_list:
        out_coff.append(a[1])
        lab_coff.append(a[2])
        x.append(a[1])
        y.append(a[2])
    temp_srcc = stats.spearmanr(out_coff, lab_coff)
    temp_plcc = stats.pearsonr(out_coff, lab_coff)
    sum_srcc += temp_srcc[0]
    sum_plcc += temp_plcc[0]
    count += 1

print(sum_srcc / count)
print(sum_plcc / count)
print(np.linalg.norm(np.array(x) - np.array(y)) / count)
srcc = stats.spearmanr(x, y)
print(srcc)
plcc = stats.pearsonr(x, y)
print(plcc)



def linear_mapping(self, pq, sq, i=0):
        if not self.mapping:
            return np.reshape(pq, (-1,))
        ones = np.ones_like(pq)
        yp1 = np.concatenate((pq, ones), axis=1)

        if self.status == 'train':
            # LSR solution of Q_i = k_1\hat{Q_i}+k_2. One can use the form of Eqn. (17) in the paper. 
            # However, for an efficient implementation, we use the matrix form of the solution here.
            # That is, h = (X^TX)^{-1}X^TY is the LSR solution of Y = Xh,
            # where X = [\hat{\mathbf{Q}}, \mathbf{1}], h = [k_1,k_2]^T, and Y=\mathbf{Q}.
            h = np.matmul(np.linalg.inv(np.matmul(yp1.transpose(), yp1)), np.matmul(yp1.transpose(), sq))
            self.k[i] = h[0].item()
            self.b[i] = h[1].item()
        else:
            h = np.reshape(np.asarray([self.k[i], self.b[i]]), (-1, 1))
        pq = np.matmul(yp1, h)

        return np.reshape(pq, (-1,))
