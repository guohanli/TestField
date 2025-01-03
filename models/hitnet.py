import torch
import torch.nn as nn
import torch.nn.functional as F
from models.pvt_v2 import pvt_v2_b2


class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding,
                              dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), "kernel size must be 3 or 7"
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


def conv(in_channels, out_channels, kernel_size, bias=False, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size, padding=(kernel_size // 2), bias=bias, stride=stride)


class CALayer(nn.Module):
    def __init__(self, channel, reduction=16, bias=False):
        super(CALayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=bias),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=bias),
            nn.Sigmoid(),
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y


class CAB(nn.Module):
    def __init__(self, n_feat, kernel_size, reduction, bias, act):
        super(CAB, self).__init__()
        modules_body = []
        modules_body.append(conv(n_feat, n_feat, kernel_size, bias=bias))
        modules_body.append(act)
        modules_body.append(conv(n_feat, n_feat, kernel_size, bias=bias))
        self.CA = CALayer(n_feat, reduction, bias=bias)
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        res = self.body(x)
        res = self.CA(res)
        res += x
        return res


class SAM(nn.Module):
    def __init__(self, ch_in=32, reduction=16):
        super(SAM, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(ch_in, ch_in // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(ch_in // reduction, ch_in, bias=False),
            nn.Sigmoid(),
        )
        self.fc_wight = nn.Sequential(
            nn.Linear(ch_in, ch_in // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(ch_in // reduction, 1, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x_h, x_l):
        b, c, _, _ = x_h.size()
        y_h = self.avg_pool(x_h).view(b, c)
        h_weight = self.fc_wight(y_h)
        y_h = self.fc(y_h).view(b, c, 1, 1)
        x_fusion_h = x_h * y_h.expand_as(x_h)
        x_fusion_h = torch.mul(x_fusion_h, h_weight.view(b, 1, 1, 1))

        b, c, _, _ = x_l.size()
        y_l = self.avg_pool(x_l).view(b, c)
        l_weight = self.fc_wight(y_l)
        y_l = self.fc(y_l).view(b, c, 1, 1)
        x_fusion_l = x_l * y_l.expand_as(x_l)
        x_fusion_l = torch.mul(x_fusion_l, l_weight.view(b, 1, 1, 1))
        x_fusion = x_fusion_h + x_fusion_l
        return x_fusion


class UpSample(nn.Module):
    def __init__(self, in_channels, s_factor):
        super(UpSample, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(in_channels + s_factor, in_channels, 1, stride=1, padding=0, bias=False)
        )

    def forward(self, x):
        x = self.up(x)
        return x


class Hitnet(nn.Module):
    def __init__(self, channel=32, n_feat=32, scale_unetfeats=32, kernel_size=3, reduction=4, bias=False,
                 act=nn.PReLU(), use_pretrain_backbone=True):
        super(Hitnet, self).__init__()
        self.backbone = pvt_v2_b2()

        if use_pretrain_backbone:
            path = "./pretrained_pvt/pvt_v2_b2.pth"
            save_model = torch.load(path)
            model_dict = self.backbone.state_dict()
            state_dict = {k: v for k, v in save_model.items() if k in model_dict.keys()}
            model_dict.update(state_dict)
            self.backbone.load_state_dict(model_dict)

        self.Translayer2_0 = BasicConv2d(64, channel, 1)
        self.Translayer2_1 = BasicConv2d(128, channel, 1)
        self.Translayer3_1 = BasicConv2d(320, channel, 1)
        self.Translayer4_1 = BasicConv2d(512, channel, 1)

        self.ca = ChannelAttention(64)
        self.sa = SpatialAttention()
        self.SAM = SAM()

        self.down05 = nn.Upsample(scale_factor=0.5, mode="bilinear", align_corners=True)
        self.out_SAM = nn.Conv2d(channel, 1, 1)
        self.out_CFM = nn.Conv2d(channel, 1, 1)

        self.decoder_level4 = [CAB(n_feat, kernel_size, reduction, bias=bias, act=act) for _ in range(2)]
        self.decoder_level3 = [CAB(n_feat + scale_unetfeats, kernel_size, reduction, bias=bias, act=act) for _ in
                               range(2)]
        self.decoder_level2 = [CAB(n_feat + (scale_unetfeats * 2), kernel_size, reduction, bias=bias, act=act) for _ in
                               range(2)]

        self.decoder_level4 = nn.Sequential(*self.decoder_level4)
        self.decoder_level3 = nn.Sequential(*self.decoder_level3)
        self.decoder_level2 = nn.Sequential(*self.decoder_level2)

        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.downsample_4 = nn.Upsample(scale_factor=0.25, mode="bilinear", align_corners=True)
        self.upsample_4 = nn.Upsample(scale_factor=4, mode="bilinear", align_corners=True)
        self.conv4 = BasicConv2d(3 * channel, channel, 3, padding=1)

        self.decoder_level1 = [CAB(64, kernel_size, reduction, bias=bias, act=act) for _ in range(2)]
        self.decoder_level1 = nn.Sequential(*self.decoder_level1)

        self.compress_out = BasicConv2d(2 * channel, channel, kernel_size=8, stride=4, padding=2)
        self.compress_out2 = BasicConv2d(2 * channel, channel, kernel_size=1)

    def forward(self, x):
        pvt = self.backbone(x)
        x1 = pvt[0]
        x2 = pvt[1]
        x3 = pvt[2]
        x4 = pvt[3]

        cim_feature = self.decoder_level1(x1)
        x2_t = self.Translayer2_1(x2)
        x3_t = self.Translayer3_1(x3)
        x4_t = self.Translayer4_1(x4)

        stage_loss = list()
        cfm_feature = None
        for iter in range(4):
            if cfm_feature == None:
                x4_t = x4_t
            else:
                x4_t = torch.cat((self.upsample_4(x4_t), cfm_feature), 1)
                x4_t = self.compress_out(x4_t)

            x4_t_feed = self.decoder_level4(x4_t)
            x3_t_feed = torch.cat((x3_t, self.upsample(x4_t_feed)), 1)
            x3_t_feed = self.decoder_level3(x3_t_feed)
            if iter > 0:
                x2_t = torch.cat((x2_t, cfm_feature), 1)
                x2_t = self.compress_out2(x2_t)
            x2_t_feed = torch.cat((x2_t, self.upsample(x3_t_feed)), 1)
            x2_t_feed = self.decoder_level2(x2_t_feed)
            cfm_feature = self.conv4(x2_t_feed)
            prediction1 = self.out_CFM(cfm_feature)
            prediction1_8 = F.interpolate(prediction1, scale_factor=8, mode="bilinear")
            stage_loss.append(prediction1_8)

        T2 = self.Translayer2_0(cim_feature)
        T2 = self.down05(T2)
        sam_feature = self.SAM(cfm_feature, T2)
        prediction2 = self.out_SAM(sam_feature)
        prediction2_8 = F.interpolate(prediction2, scale_factor=8, mode="bilinear")
        return stage_loss, prediction2_8


if __name__ == "__main__":
    model = Hitnet().cuda()
    input_tensor = torch.randn(1, 3, 352, 352).cuda()
    prediction1, prediction2 = model(input_tensor)
    print(prediction1, prediction2)