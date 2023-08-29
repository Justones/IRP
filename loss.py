import torch
import torch.nn as nn
import torch.nn.functional as F

def L1_loss(out, gt):
    return torch.abs(out-gt).mean()

def L2_loss(out, gt):
    mse_loss = torch.nn.MSELoss()
    return mse_loss(out, gt)

def smooth_L1(out, gt):
    smooth_loss = torch.nn.SmoothL1Loss()
    return smooth_loss(out, gt)
