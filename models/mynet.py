import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.pvt_v2 import pvt_v2_b2


# Channel Attention Block
class CAB(nn.Module):
    def __init__(self, channels, ratio=16):
        super(CAB, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        # Combine FC layers to reduce parameters
        mid_channels = max(8, channels // ratio)
        self.shared_mlp = nn.Sequential(
            nn.Conv2d(channels, mid_channels, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.shared_mlp(self.avg_pool(x))
        max_out = self.shared_mlp(self.max_pool(x))
        return self.sigmoid(avg_out + max_out)


# Spatial Attention Block
class SAB(nn.Module):
    def __init__(self, kernel_size=7):
        super(SAB, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out = torch.max(x, dim=1, keepdim=True)[0]
        return self.sigmoid(self.conv(torch.cat([avg_out, max_out], dim=1)))


# Multi-scale Depth-wise Convolution
class MSDC(nn.Module):
    def __init__(self, channels, kernel_sizes=[1, 3], stride=1):
        super(MSDC, self).__init__()
        self.convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(channels, channels, k, stride, padding=k // 2, groups=channels, bias=False),
                nn.BatchNorm2d(channels),
                nn.ReLU6(inplace=True)
            ) for k in kernel_sizes
        ])

    def forward(self, x):
        return [conv(x) for conv in self.convs]


# Multi-scale Convolution Block
class MSCB(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_sizes=[1, 3], expansion_factor=2):
        super(MSCB, self).__init__()
        ex_channels = int(in_channels * expansion_factor)

        self.expand_conv = nn.Sequential(
            nn.Conv2d(in_channels, ex_channels, 1, bias=False),
            nn.BatchNorm2d(ex_channels),
            nn.ReLU6(inplace=True)
        )

        self.msdc = MSDC(ex_channels, kernel_sizes)

        self.project_conv = nn.Sequential(
            nn.Conv2d(ex_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        x = self.expand_conv(x)
        outputs = self.msdc(x)
        out = outputs[0]
        for i in range(1, len(outputs)):
            out = out + outputs[i]
        return self.project_conv(out)


# Efficient Multi-scale Convolutional Attention Module
class EMCAM(nn.Module):
    def __init__(self, channels=64, kernel_sizes=[1, 3], expansion_factor=2):
        super(EMCAM, self).__init__()
        self.cab = CAB(channels)
        self.sab = SAB()
        self.mscb = MSCB(channels, channels, kernel_sizes, expansion_factor)

    def forward(self, x):
        x = x * self.cab(x)
        x = x * self.sab(x)
        return self.mscb(x)


class MyNet(nn.Module):
    def __init__(self):
        super(MyNet, self).__init__()

        # Load backbone
        current_dir = os.path.dirname(os.path.abspath(__file__))
        pretrained_path = os.path.join(os.path.dirname(current_dir), 'pretrained_pvt', 'pvt_v2_b2.pth')

        self.backbone = pvt_v2_b2()
        if os.path.exists(pretrained_path):
            state_dict = torch.load(pretrained_path, weights_only=True)
            self.backbone.load_state_dict({k: v for k, v in state_dict.items()
                                           if k in self.backbone.state_dict()})

        # Channel reduction layers
        self.cr = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(c, 64, 1, bias=False),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True)
            ) for c in [512, 320, 128, 64]
        ])

        # EMCAM modules
        self.emcam = nn.ModuleList([
            EMCAM(64, kernel_sizes=[1, 3], expansion_factor=2)
            for _ in range(4)
        ])

        # Additional convolution
        self.conv4 = nn.Sequential(
            nn.Conv2d(64, 64, 3, 1, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        # Refinement layers
        self.refine = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(64, 64, 3, 1, 1, bias=False),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True)
            ) for _ in range(4)
        ])

        # Output layers
        self.map = nn.Conv2d(64, 1, 7, 1, 3)
        self.out_map = nn.Conv2d(64, 1, 7, 1, 3)

    def forward(self, x):
        # Extract features from backbone
        features = self.backbone(x)
        x1, x2, x3, x4 = features

        # Channel reduction
        reduced_features = [cr(f) for cr, f in zip(self.cr, [x4, x3, x2, x1])]
        x_4, x_3, x_2, x_1 = reduced_features

        stage_loss1 = []
        stage_loss2 = []

        # Initialize multi-scale feedback with correct sizes
        feedback_features = {
            'f1': None,  # 1/4 scale
            'f2': None,  # 1/8 scale
            'f3': None,  # 1/16 scale
            'f4': None  # 1/32 scale
        }

        for iter in range(3):
            # Apply multi-scale feedback
            f_1 = x_1 + feedback_features['f1'] if feedback_features['f1'] is not None else x_1
            f_2 = x_2 + feedback_features['f2'] if feedback_features['f2'] is not None else x_2
            f_3 = x_3 + feedback_features['f3'] if feedback_features['f3'] is not None else x_3
            f_4 = x_4 + feedback_features['f4'] if feedback_features['f4'] is not None else x_4

            # Global feature extraction path (bottom-up)
            gf0 = self.emcam[0](f_1)
            gf0 = F.interpolate(gf0, size=x_2.shape[2:], mode='bilinear', align_corners=False)

            gf1 = self.emcam[1](gf0 + f_2)
            gf1 = F.interpolate(gf1, size=x_3.shape[2:], mode='bilinear', align_corners=False)

            gf2 = self.emcam[2](gf1 + f_3)
            gf2 = F.interpolate(gf2, size=x_4.shape[2:], mode='bilinear', align_corners=False)

            gf3 = self.emcam[3](gf2 + f_4)
            gf3 = self.conv4(gf3)
            gf_pre = self.map(gf3)

            # Refinement path (top-down)
            # Store intermediate features for feedback with correct sizes
            rf4_temp = gf3 + f_4
            rf4 = self.refine[0](rf4_temp)
            rf4_feedback = rf4.clone()  # Store before interpolation
            rf4 = F.interpolate(rf4, size=x_3.shape[2:], mode='bilinear', align_corners=False)

            rf3_temp = rf4 + f_3
            rf3 = self.refine[1](rf3_temp)
            rf3_feedback = rf3.clone()  # Store before interpolation
            rf3 = F.interpolate(rf3, size=x_2.shape[2:], mode='bilinear', align_corners=False)

            rf2_temp = rf3 + f_2
            rf2 = self.refine[2](rf2_temp)
            rf2_feedback = rf2.clone()  # Store before interpolation
            rf2 = F.interpolate(rf2, size=x_1.shape[2:], mode='bilinear', align_corners=False)

            rf1_temp = rf2 + f_1
            rf1 = self.refine[3](rf1_temp)
            rf1_feedback = rf1.clone()  # Store before interpolation

            rf1_map = self.out_map(rf1)

            # Update multi-scale feedback with correctly sized features
            feedback_features['f4'] = rf4_feedback  # 1/32 scale feedback
            feedback_features['f3'] = rf3_feedback  # 1/16 scale feedback
            feedback_features['f2'] = rf2_feedback  # 1/8 scale feedback
            feedback_features['f1'] = rf1_feedback  # 1/4 scale feedback

            # Resize predictions to match input size
            gf_pre = F.interpolate(gf_pre, size=x.shape[2:], mode='bilinear', align_corners=False)
            rf1_map = F.interpolate(rf1_map, size=x.shape[2:], mode='bilinear', align_corners=False)

            stage_loss1.append(gf_pre)
            stage_loss2.append(rf1_map)

        return stage_loss1, stage_loss2


if __name__ == '__main__':
    from utils.tools import get_model_summary

    gpu_id = 1
    device = f'cuda:{gpu_id}'
    model = MyNet().to(device)

    # 获取并打印模型统计信息
    summary = get_model_summary(model, device)
    print(summary)

    # 推理测试
    input_tensor = torch.randn(8, 3, 704, 704).to(device)

    prediction1, prediction2 = model(input_tensor)
