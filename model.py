import torch
import torch.nn as nn
import pretrainedmodels
from torch.nn import init
from utils import load_checkpoint
import torch.nn.functional as F
from guided_filter_pytorch.guided_filter import GuidedFilter


class IPR(torch.nn.Module):

    def __init__(self, multitask=0):
        """Declare all needed layers."""
        super(IRP, self).__init__()

        # Convolution and pooling layers of VGG-16.
        # self.saliency = ResNet50_Saliency()

        self.blur = pretrainedmodels.resnet50(num_classes=1000, pretrained='imagenet')
        self.blur = nn.Sequential(*list(self.blur.children())[:-2])

        # self.blur = pretrainedmodels.resnet50(num_classes=1000, pretrained='imagenet')
        # self.blur = nn.Sequential(*list(self.blur.children())[:-2])
        self.noise = Noise_branch(512)

        self.hist = Hist_branch()

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc_blur = nn.Sequential(
            nn.Conv2d(2048, 512, 1),
        )
        self.fc_hist = nn.Sequential(
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
        )

        self.skff1 = SKFF(512)
        self.skff2 = SKFF(512)
        self.skff3 = SKFF(512)

        self.fc_all = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

        self.fc_all.apply(weights_init_xavier)
        self.fc_hist.apply(weights_init_xavier)
        self.fc_blur.apply(weights_init_xavier)

        self.sigmoid = nn.Sigmoid()

    def forward(self, X, X_exp):
        """Forward pass of the network.
        """
        N, C, W, H = X.size()

        X_Guided = GuidedFilter(5, 0.01)(X_exp, X_exp)

        X_blur = self.fc_blur(self.blur(X_Guided))
        X_hist = (self.fc_hist(self.hist(X))).unsqueeze(-1).unsqueeze(-1)
        X_hist = X_hist.expand_as(X_blur)

        # X_blur = self.fc_blur(self.pool(self.blur(X_Guided)).view(N, -1))
        X_noise = F.interpolate(self.noise(X_exp), scale_factor=1/4)

        X_hist, X_blur, X_noise = self.skff1([X_hist, X_blur, X_noise])
        X_hist, X_blur, X_noise = self.skff2([X_hist, X_blur, X_noise])
        X_hist, X_blur, X_noise = self.skff3([X_hist, X_blur, X_noise])
        X_hist = self.pool(X_hist).view(N, -1)
        X_blur = self.pool(X_blur).view(N, -1)
        X_noise = self.pool(X_noise).view(N, -1)

        X = self.fc_all(X_hist+X_blur+X_noise)

        return X


class Hist_branch(nn.Module):
    def __init__(self):
        super(Hist_branch, self).__init__()
        self.conv = nn.Sequential(nn.Conv1d(1, 1, 7, padding=3), nn.ReLU(), nn.Conv1d(1, 1, 7, padding=3), nn.ReLU())
        self.fc = nn.Sequential(nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, 256), nn.ReLU())

    def forward(self, x):
        N = x.shape[0]
        hist_x = torch.histc(x[0, :, :, :], bins=256, min=0, max=1).view(1, -1).float()/(x.shape[2]*x.shape[3])
        for i in range(1, N):
            temp_x = torch.histc(x[i, :, :, :], bins=256, min=0, max=1).view(1, -1).float()/(x.shape[2]*x.shape[3])
            # temp_x = (temp_x - temp_x.max()) / (temp_x.max() - temp_x.min() + 1e-9)
            hist_x = torch.cat((hist_x, temp_x), dim=0)
        hist_x = hist_x.view(N, 1, -1).float() * 200
        conv = self.conv(hist_x).view(N, -1)
        # print(conv.shape)
        out = self.fc(conv)
        return out


class Noise_branch(torch.nn.Module):

    def __init__(self, in_chn):
        """Declare all needed layers."""
        super(Noise_branch, self).__init__()

        self.blur = pretrainedmodels.resnet50(num_classes=1000, pretrained='imagenet')
        self.blur = nn.Sequential(*list(self.blur.children())[:-2])

        self.dilate = ASPP(in_chn, in_chn)
        self.dilate2 = ASPP(in_chn, in_chn)
        self.dilate3 = ASPP(in_chn, in_chn)

    def forward(self, X):
        """Forward pass of the network.
        """
        N, C, W, H = X.size()

        for ii, model in enumerate(self.blur):
            X = model(X)
            if ii == 5:
                break

        X = self.dilate(X)
        X = self.dilate2(X)
        X = self.dilate3(X)

        return X


class ASPP(nn.Module):
    def __init__(self, in_channel, out_channel, downsample=False):
        super(ASPP, self).__init__()
        # self.mean = nn.AdaptiveAvgPool2d((1, 1))  # (1,1)means ouput_dim
        # self.conv = nn.Sequential(nn.Conv2d(in_channel, depth, 1, 1),
        #                           nn.BatchNorm2d(depth))
        self.downsample = downsample

        self.atrous_block1 = nn.Conv2d(in_channel, in_channel//2, 3, 1, padding=1)
        self.atrous_block2 = nn.Conv2d(in_channel, in_channel//2, 3, 1, padding=2, dilation=2)
        self.atrous_block3 = nn.Conv2d(in_channel, in_channel//2, 3, 1, padding=3, dilation=3)
        self.atrous_block4 = nn.Conv2d(in_channel, in_channel//2, 3, 1, padding=5, dilation=5)

        self.conv_1x1_output = nn.Conv2d(in_channel*2, out_channel, 1, 1)

        self.ca = ca_layer(out_channel)

        self.down = nn.MaxPool2d(2, 2)

        self.atrous_block1.apply(weights_init_xavier)
        self.atrous_block2.apply(weights_init_xavier)
        self.atrous_block3.apply(weights_init_xavier)
        self.atrous_block4.apply(weights_init_xavier)
        self.conv_1x1_output.apply(weights_init_xavier)

    def forward(self, x):

        atrous_block1 = self.atrous_block1(x)
        atrous_block2 = self.atrous_block2(x)
        atrous_block3 = self.atrous_block3(x)
        atrous_block4 = self.atrous_block4(x)

        out = self.conv_1x1_output(torch.cat((atrous_block1, atrous_block2,
                                              atrous_block3, atrous_block4), dim=1))
        out = self.ca(out)

        if self.downsample:
            out = self.down(out)

        return out


class ca_layer(nn.Module):
    def __init__(self, channel, reduction=4, bias=True):
        super(ca_layer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
                nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=bias),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=bias),
                nn.Sigmoid()
        )
        self.conv_du.apply(weights_init_xavier)

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return y*x


class SKFF(nn.Module):
    def __init__(self, in_channels, height=3, reduction=4, bias=False):
        super(SKFF, self).__init__()

        self.height = height
        d = max(int(in_channels / reduction), 4)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_du = nn.Sequential(nn.Conv2d(in_channels, d, 1, padding=0, bias=bias), nn.PReLU())

        self.fcs = nn.ModuleList([])
        for i in range(self.height):
            self.fcs.append(nn.Conv2d(d, in_channels, kernel_size=1, stride=1, bias=bias))
            # self.fcs[i].apply(weights_init_xavier)

        self.softmax = nn.Softmax(dim=1)

        self.fc_dis = nn.Sequential(
                nn.Linear(d, d),
                nn.ReLU(),
                nn.Linear(d, d),
                nn.ReLU(),
                nn.Linear(d, 5),
            )

        self.conv_du.apply(weights_init_xavier)
        self.fcs.apply(weights_init_xavier)
        self.fc_dis.apply(weights_init_xavier)

    def forward(self, inp_feats):
        batch_size = inp_feats[0].shape[0]
        n_feats = inp_feats[0].shape[1]

        inp_feats = torch.cat(inp_feats, dim=1)
        inp_feats = inp_feats.view(batch_size, self.height, n_feats, inp_feats.shape[2], inp_feats.shape[3])

        feats_U = torch.sum(inp_feats, dim=1)
        feats_S = self.avg_pool(feats_U)
        feats_Z = self.conv_du(feats_S)

        attention_vectors = [fc(feats_Z) for fc in self.fcs]
        attention_vectors = torch.cat(attention_vectors, dim=1)
        attention_vectors = attention_vectors.view(batch_size, self.height, n_feats, 1, 1)
        # stx()
        attention_vectors = self.softmax(attention_vectors)

        # feats_V = torch.sum(inp_feats * attention_vectors, dim=1)
        feats_V = inp_feats * attention_vectors
        # feats_V.permute(1,0,2,3,4)

        tensor = []
        for i in range(self.height):
            tensor.append(feats_V[:, i, :, :, :])
        #
        # dist = self.fc_dis(feats_Z.view(batch_size, -1))

        return tensor


def calc_mean_std(feat, eps=1e-5):
    # eps is a small value added to the variance to avoid divide-by-zero.
    size = feat.size()
    assert (len(size) == 4)
    N, C = size[:2]
    feat_var = feat.view(N, C, -1).var(dim=2) + eps
    feat_std = feat_var.sqrt().view(N, C, 1, 1)
    feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
    return feat_mean, feat_std


def weights_init_xavier(m):
    classname = m.__class__.__name__
    # print(classname)
    # if isinstance(m, nn.Conv2d):
    if classname.find('Conv') != -1:
        init.kaiming_normal_(m.weight.data)
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data)
    elif classname.find('BatchNorm2d') != -1:
        init.uniform_(m.weight.data, a=0.98, b=1.02)
        init.constant_(m.bias.data, 0.0)
