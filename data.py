from PIL import Image
import torch
import torchvision.transforms as TF
import numpy as np
from glob import glob
import os
from torch.utils.data import Dataset


class DataSet(Dataset):
    def __init__(self, phase ='train', path='./data_new/temp_new_test/'):
        super(DataSet,self)

        self.imgs = []
        self.labels = []
        self.pixel_wise_labels = []
        self.phase = phase

        self.ratio_list = [1, 1.5, 2, 2.5, 3, 5, 10, 20, 30, 50, 75]
        list_fp = open('./data/list_index.txt')
        string_num = list_fp.readline().split(',')
        list_index = [int(temp) for temp in string_num]
        list_fp.close()
        #list_index = int(string_num.split(','))
        #print(len(list_index))
        #print(list_index)
        self.ratios = []

        idx1 = 350
        idx2 = 400
        idx3 = 500
        
        if phase == 'train':
            start_idx = 0
            end_idx = idx1
        elif phase == 'validation':
            start_idx = idx1
            end_idx = idx2
        elif phase == 'test':
            start_idx = idx2
            end_idx = idx3


        for ratio in self.ratio_list:
            fp = open('/home/ssl/work/codes/IRP/data_new/IRP_label/' + str(ratio) + '_metric.txt')
            imgs_path = sorted(glob(path + 'scene*/' + str(ratio) + '.png'))
            pixel_wise_label_path = sorted(glob())
            temp_labels = [np.float32(line) for line in fp.readlines()]
            self.labels = self.labels + [np.float32(temp_labels[idx]*10) for idx in list_index[start_idx:end_idx]]
            self.imgs = self.imgs + [imgs_path[idx] for idx in list_index[start_idx:end_idx]]
            self.pixel_wise_labels = self.pixel_wise_labels + [pixel_wise_label_path[idx] for idx in list_index[start_idx:end_idx]]
            self.ratios = self.ratios + [ratio for idx in range(start_idx, end_idx)]
            '''
            if phase == 'train':
                imgs_path = sorted(glob(path + 'scene*/' + str(ratio) + '.png'))
                pixel_wise_label_path = sorted(glob())
                temp_labels = [np.float32(line) for line in fp.readlines()]
                
                self.labels = self.labels + [np.float32(temp_labels[idx]*10) for idx in list_index[0:idx1]]
                self.imgs = self.imgs + [imgs_path[idx] for idx in list_index[0:idx1]]
                self.pixel_wise_labels = self.pixel_wise_labels + [pixel_wise_label_path[idx] for idx in list_index[0:idx1]]
                self.ratios = self.ratios + [ratio for idx in range(0, idx1)]
            elif phase == 'validation':
                imgs_path = sorted(glob(path + 'scene*/' + str(ratio) + '.png'))[0:idx3]
                self.imgs = self.imgs + [imgs_path[idx] for idx in list_index[idx1:idx2]]
                temp_labels = [np.float32(line) for line in fp.readlines()]
                self.labels = self.labels + [np.float32(temp_labels[idx]*10) for idx in list_index[idx1:idx2]]

                self.pixel_wise_labels = self.pixel_wise_labels + [pixel_wise_label_path[idx] for idx in list_index[idx1:idx2]]
                self.ratios = self.ratios + [ratio for idx in range(idx1, idx2)]
            elif phase == 'test':
                imgs_path = sorted(glob(path + 'scene*/' + str(ratio) + '.png'))[0:idx3]
                self.imgs = self.imgs + [imgs_path[idx] for idx in list_index[idx2:idx3]]
                temp_labels = [np.float32(line) for line in fp.readlines()]
                self.labels = self.labels + [np.float32(temp_labels[idx] * 10) for idx in list_index[idx2:idx3]]
                self.pixel_wise_labels = self.pixel_wise_labels + [pixel_wise_label_path[idx] for idx in list_index[idx2:idx3]]
                self.ratios = self.ratios + [ratio for idx in range(idx2, idx3)]
            '''
            fp.close()

        self.len = len(self.labels)
        if phase == 'train':
            self.transforms = TF.Compose([
                    TF.Normalize(mean=(0.485, 0.456, 0.406),
                                std=(0.229, 0.224, 0.225))
                ])
        else:
            self.transforms = TF.Compose([
                    TF.Normalize(mean=(0.485, 0.456, 0.406),
                                std=(0.229, 0.224, 0.225))
                ])
        print(self.len)

    def __getitem__(self, index):

        label = self.labels[index]
        pixel_label = self.pixel_wise_labels[index]
        if self.phase == 'train':
            img = pil_loader(self.imgs[index])
            img = TF.RandomHorizontalFlip(p=0.5)(img)
            img = TF.ToTensor()(img)
            img_exp = torch.pow((torch.pow(img, 2.2) * self.ratios[index]), 1/2.2)
            return img, self.transforms(img_exp), label, self.imgs[index]

        else:
            img = pil_loader(self.imgs[index])
            img = TF.ToTensor()(img)
            img_exp = torch.pow((torch.pow(img, 2.2) * self.ratios[index]), 1/2.2)
            return img, self.transforms(img_exp), label, self.imgs[index]

    def __len__(self):
        return self.len

def pil_loader(path):
    # print(path)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')
