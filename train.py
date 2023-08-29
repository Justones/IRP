
import numpy as np
from model import *
from data import DataSet
from torch.utils.data import DataLoader
from loss import L2_loss
from utils import save_checkpoint, load_checkpoint, make_path, token_cmp, token_cmp1
import os
from scipy import stats

import random

config = {
    'data_path': './data_new/temp_new_test/',
    'save_path': './results/',
    'lr': 1e-4,
    'epoch': 30,
    'batch_size': 16,
    'patch_size': 256,
    'ckpt': './ckpt/',
    'ratio': 'test2',
    'result_path': './results/',
    'Resume': False,
}


class Trainer(object):
    
    def __init__(self, config):
        
        '''
        model initialize
        '''

        seed = int(19920318)
        torch.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        np.random.seed(seed)
        random.seed(seed)

        self.model = IRP()
        self.model = self.model.cuda()

        self.optimizer = torch.optim.Adam(self.model.parameters(), config['lr'])

        self.start_epoch = 0

        if config['Resume']:
            # load_checkpoint(self.model, self.optimizer, self.start_epoch, config['ckpt'])
            checkpoint = torch.load(os.path.join(config['ckpt'], config['ratio'], 'epoch_30_join.pt'))
            self.model.load_state_dict(checkpoint['state_dict'])

        self.epoch = config['epoch']

        self.model_save_dir = os.path.join(config['ckpt'], str(config['ratio']))
        make_path(self.model_save_dir)
        #if not os.path.exists(self.model_save_dir):
        #    os.makedirs(self.model_save_dir)
        
        self.batch_size = config['batch_size']
        dataset = DataSet(phase='train', path= config['data_path'])
        self.train_data_loader = DataLoader(dataset, num_workers = 8, batch_size = config['batch_size'], shuffle = True)

        test_dataset = DataSet(phase='test', path=config['data_path'])
        self.test_data_loader = DataLoader(test_dataset, num_workers=8, batch_size=1, shuffle=True)

        val_dataset = DataSet(phase='validation', path=config['data_path'])
        self.val_data_loader = DataLoader(val_dataset, num_workers=8, batch_size=1, shuffle=True)
        
    def train(self, epoch):
        self.model.train()
        sum_loss = 0
        cnt = 0
        
        for inp_img, inp_img_exp, label, img_name in self.train_data_loader:
            inp_img, inp_img_exp, label = inp_img.cuda(), inp_img_exp.cuda(), label.cuda()
            label = label.view(-1, 1)

            # train model
            out = self.model(inp_img, inp_img_exp)

            loss = L2_loss(out, label)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            cnt += 1
            sum_loss = sum_loss + float(loss)

        print('epoch: %d | loss: %f'% (epoch, sum_loss/cnt))
    
    def validation(self):
        self.model.eval()

        all_list = []

        for inp_img, inp_img_exp, label, path in self.val_data_loader:
            inp_img, inp_img_exp, label = inp_img.cuda(), inp_img_exp.cuda(), label.cuda()
            q = self.model(inp_img, inp_img_exp)

            q = q.detach()

            for idx in range(len(label)):
                all_list.append([path[idx], float(q[idx]), float(label[idx])])

        # print(all_list)
        all_list.sort(key=token_cmp)
        x = []
        y = []

        for idx in range(0, len(all_list), 11):
            temp_list = all_list[idx:idx + 11]
            temp_list.sort(key=token_cmp1)

            for a in temp_list:
                x.append(a[1])
                y.append(a[2])

        srcc = stats.spearmanr(x, y)
        return srcc


    def test(self):
        self.model.eval()

        all_list = []

        for inp_img, inp_img_exp, label, path in self.test_data_loader:
            inp_img, inp_img_exp, label = inp_img.cuda(), inp_img_exp.cuda(), label.cuda()
            q = self.model(inp_img, inp_img_exp)

            q = q.detach()

            for idx in range(len(label)):
                all_list.append([path[idx], float(q[idx]), float(label[idx])])

        # print(all_list)
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

    def save(self,epoch):
        save_path = os.path.join(self.model_save_dir, 'epoch_{}_IRP.pt'.format(epoch))
        save_checkpoint(self.model, self.optimizer, epoch, save_path)

    def start(self):

        val_srcc_top = 0.

        for epoch in range(self.start_epoch+1, self.epoch+1):

            self.train(epoch)
            val_srcc = self.validation()
            if val_srcc > val_srcc_top:
                self.test()
                val_srcc_top = val_srcc
                self.save(epoch)

            if epoch == 10:
                config['lr'] = config['lr'] / 10
                self.optimizer = torch.optim.Adam(self.model.parameters(), config['lr'])

            if epoch == 20:
                config['lr'] = config['lr'] / 10
                self.optimizer = torch.optim.Adam(self.model.parameters(), config['lr'])

            if epoch == 30:
                config['lr'] = config['lr'] / 10
                self.optimizer = torch.optim.Adam(self.model.parameters(), config['lr'])
        
if __name__ =='__main__':
    print(config)
    trainer = Trainer(config)
    trainer.start()
    
