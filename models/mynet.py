import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.pvt_v2 import pvt_v2_b2


class Classifier_Module(nn.Module):
    """多尺度分类器模块，使用不同膨胀率的卷积进行特征提取"""

    def __init__(self, dilation_series, padding_series, num_classes):
        super(Classifier_Module, self).__init__()
        self.conv2d_list = nn.ModuleList()
        for dilation, padding in zip(dilation_series, padding_series):
            # 对每个膨胀率创建一个卷积层
            self.conv2d_list.append(
                nn.Conv2d(64, num_classes, kernel_size=3, stride=1, padding=padding, dilation=dilation, bias=True))

        # 初始化权重
        for m in self.conv2d_list:
            m.weight.data.normal_(0, 0.01)

    def forward(self, x):
        # 将所有不同膨胀率的卷积结果相加
        out = self.conv2d_list[0](x)
        for i in range(len(self.conv2d_list) - 1):
            out += self.conv2d_list[i + 1](x)
        return out

    def initialize(self):
        pass


class Context_Exploration_Block(nn.Module):
    """上下文探索模块，通过多分支结构捕获多尺度上下文信息"""

    def __init__(self, input_channels):
        super(Context_Exploration_Block, self).__init__()
        self.input_channels = input_channels
        self.channels_single = int(input_channels / 4)

        # 通道降维模块，将输入通道数减少为1/4
        self.p1_channel_reduction = nn.Sequential(
            nn.Conv2d(self.input_channels, self.channels_single, 1, 1, 0),
            nn.BatchNorm2d(self.channels_single), nn.ReLU())
        self.p2_channel_reduction = nn.Sequential(
            nn.Conv2d(self.input_channels, self.channels_single, 1, 1, 0),
            nn.BatchNorm2d(self.channels_single), nn.ReLU())
        self.p3_channel_reduction = nn.Sequential(
            nn.Conv2d(self.input_channels, self.channels_single, 1, 1, 0),
            nn.BatchNorm2d(self.channels_single), nn.ReLU())
        self.p4_channel_reduction = nn.Sequential(
            nn.Conv2d(self.input_channels, self.channels_single, 1, 1, 0),
            nn.BatchNorm2d(self.channels_single), nn.ReLU())

        # 四个平行分支，每个分支包含普通卷积和空洞卷积
        # 分支1：1x1卷积 + 3x3空洞卷积(rate=1)
        self.p1 = nn.Sequential(
            nn.Conv2d(self.channels_single, self.channels_single, 1, 1, 0),
            nn.BatchNorm2d(self.channels_single), nn.ReLU())
        self.p1_dc = nn.Sequential(
            nn.Conv2d(self.channels_single, self.channels_single, kernel_size=3, stride=1, padding=1, dilation=1),
            nn.BatchNorm2d(self.channels_single), nn.ReLU())

        # 分支2：3x3卷积 + 3x3空洞卷积(rate=2)
        self.p2 = nn.Sequential(
            nn.Conv2d(self.channels_single, self.channels_single, 3, 1, 1),
            nn.BatchNorm2d(self.channels_single), nn.ReLU())
        self.p2_dc = nn.Sequential(
            nn.Conv2d(self.channels_single, self.channels_single, kernel_size=3, stride=1, padding=2, dilation=2),
            nn.BatchNorm2d(self.channels_single), nn.ReLU())

        # 分支3：5x5卷积 + 3x3空洞卷积(rate=4)
        self.p3 = nn.Sequential(
            nn.Conv2d(self.channels_single, self.channels_single, 5, 1, 2),
            nn.BatchNorm2d(self.channels_single), nn.ReLU())
        self.p3_dc = nn.Sequential(
            nn.Conv2d(self.channels_single, self.channels_single, kernel_size=3, stride=1, padding=4, dilation=4),
            nn.BatchNorm2d(self.channels_single), nn.ReLU())

        # 分支4：7x7卷积 + 3x3空洞卷积(rate=8)
        self.p4 = nn.Sequential(
            nn.Conv2d(self.channels_single, self.channels_single, 7, 1, 3),
            nn.BatchNorm2d(self.channels_single), nn.ReLU())
        self.p4_dc = nn.Sequential(
            nn.Conv2d(self.channels_single, self.channels_single, kernel_size=3, stride=1, padding=8, dilation=8),
            nn.BatchNorm2d(self.channels_single), nn.ReLU())

        # 最终特征融合
        self.fusion = nn.Sequential(nn.Conv2d(self.input_channels, self.input_channels, 1, 1, 0),
                                    nn.BatchNorm2d(self.input_channels), nn.ReLU())

    def forward(self, x):
        """
        Args:
            x: 输入特征图 (B, C, H, W)
        Returns:
            ce: 融合后的特征图 (B, C, H, W)
        """
        # 分支1处理
        p1_input = self.p1_channel_reduction(x)
        p1 = self.p1(p1_input)
        p1_dc = self.p1_dc(p1)

        # 分支2处理，加入分支1的空洞卷积结果
        p2_input = self.p2_channel_reduction(x) + p1_dc
        p2 = self.p2(p2_input)
        p2_dc = self.p2_dc(p2)

        # 分支3处理，加入分支2的空洞卷积结果
        p3_input = self.p3_channel_reduction(x) + p2_dc
        p3 = self.p3(p3_input)
        p3_dc = self.p3_dc(p3)

        # 分支4处理，加入分支3的空洞卷积结果
        p4_input = self.p4_channel_reduction(x) + p3_dc
        p4 = self.p4(p4_input)
        p4_dc = self.p4_dc(p4)

        # 融合所有分支的空洞卷积结果
        ce = self.fusion(torch.cat((p1_dc, p2_dc, p3_dc, p4_dc), 1))

        return ce


class Focus(nn.Module):
    """焦点模块：融合不同层级的特征并进行细化"""

    def __init__(self, channel1, channel2):
        super(Focus, self).__init__()
        self.channel1 = channel1  # 当前层级的通道数
        self.channel2 = channel2  # 高层级的通道数

        # 上采样和特征处理模块
        self.up = nn.Sequential(nn.Conv2d(self.channel2, self.channel1, 7, 1, 3),
                                nn.BatchNorm2d(self.channel1), nn.ReLU(),
                                nn.UpsamplingBilinear2d(scale_factor=2))
        self.up2 = nn.Sequential(nn.Conv2d(self.channel2, self.channel1, 7, 1, 3),
                                 nn.BatchNorm2d(self.channel1), nn.ReLU())

        # 预测图处理
        self.input_map = nn.Sequential(nn.UpsamplingBilinear2d(scale_factor=2), nn.Sigmoid())
        self.input_map2 = nn.Sequential(nn.Sigmoid())
        self.output_map = nn.Conv2d(self.channel1, 1, 7, 1, 3)

        # 上下文探索模块
        self.fp = Context_Exploration_Block(self.channel1)  # 处理假阳性特征
        self.fn = Context_Exploration_Block(self.channel1)  # 处理假阴性特征

        # 可学习参数
        self.alpha = nn.Parameter(torch.ones(1))  # 控制假阳性特征的权重
        self.beta = nn.Parameter(torch.ones(1))  # 控制假阴性特征的权重

        # 归一化和激活层
        self.bn1 = nn.BatchNorm2d(self.channel1)
        self.relu1 = nn.ReLU()
        self.bn2 = nn.BatchNorm2d(self.channel1)
        self.relu2 = nn.ReLU()

    def forward(self, x, y, in_map, upsample=True):
        """
        Args:
            x: 当前层级特征 (B, C, H, W)
            y: 高层级特征 (B, C, H/2, W/2)
            in_map: 高层级预测图 (B, 1, H/2, W/2)
            upsample: 是否需要上采样
        """
        # 处理高层级特征
        if upsample:
            up = self.up(y)  # 上采样并处理
            input_map = self.input_map(in_map)  # 预测图上采样
        else:
            up = self.up2(y)  # 只处理不上采样
            input_map = self.input_map2(in_map)

        # 提取假阳性和假阴性特征
        f_feature = x * input_map  # 假阳性特征
        b_feature = x * (1 - input_map)  # 假阴性特征

        # 通过上下文探索模块处理特征
        fp = self.fp(f_feature)  # 处理假阳性特征
        fn = self.fn(b_feature)  # 处理假阴性特征

        # 特征细化
        refine1 = up - (self.alpha * fp)  # 减去加权的假阳性特征
        refine1 = self.bn1(refine1)
        refine1 = self.relu1(refine1)

        refine2 = refine1 + (self.beta * fn)  # 加上加权的假阴性特征
        refine2 = self.bn2(refine2)
        refine2 = self.relu2(refine2)

        # 生成输出预测图
        output_map = self.output_map(refine2)

        return refine2, output_map


class Att(nn.Module):
    """注意力模块：结合局部和全局注意力"""

    def __init__(self, channels=64, r=4):
        super(Att, self).__init__()
        out_channels = int(channels // r)

        # 局部注意力分支：使用1x1卷积
        self.local_att = nn.Sequential(
            nn.Conv2d(channels, out_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels)
        )

        # 全局注意力分支：使用全局平均池化和1x1卷积
        self.global_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, out_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels)
        )

        self.sig = nn.Sigmoid()

    def forward(self, x):
        """
        Args:
            x: 输入特征图 (B, C, H, W)
        Returns:
            wei: 注意力权重图 (B, C, H, W)
        """
        xl = self.local_att(x)  # 局部注意力
        xg = self.global_att(x)  # 全局注意力
        xlg = xl + xg  # 融合局部和全局注意力
        wei = self.sig(xlg)  # 生成权重图

        return wei

    def initialize(self):
        pass


class MyNet(nn.Module):
    """主网络架构"""

    def __init__(self):
        super(MyNet, self).__init__()

        # 设置预训练权重路径
        current_dir = os.path.dirname(os.path.abspath(__file__))
        parent_dir = os.path.dirname(current_dir)
        pretrained_path = os.path.join(parent_dir, 'pretrained_pvt', 'pvt_v2_b2.pth')

        # 加载预训练的PVT-v2主干网络
        self.backbone = pvt_v2_b2()
        save_model = torch.load(pretrained_path)
        model_dict = self.backbone.state_dict()
        state_dict = {k: v for k, v in save_model.items() if k in model_dict.keys()}
        model_dict.update(state_dict)
        self.backbone.load_state_dict(model_dict)

        # 通道调整模块：统一不同层级特征的通道数为64
        self.cr4 = nn.Sequential(nn.Conv2d(512, 64, 1), nn.BatchNorm2d(64), nn.ReLU())
        self.cr3 = nn.Sequential(nn.Conv2d(320, 64, 1), nn.BatchNorm2d(64), nn.ReLU())
        self.cr2 = nn.Sequential(nn.Conv2d(128, 64, 1), nn.BatchNorm2d(64), nn.ReLU())
        self.cr1 = nn.Sequential(nn.Conv2d(64, 64, 1), nn.BatchNorm2d(64), nn.ReLU())

        # 多尺度分类器模块
        self.conv0 = nn.Sequential(self._make_pred_layer(Classifier_Module, [1, 6, 8, 12], [1, 6, 8, 12], 64),
                                   nn.BatchNorm2d(64), nn.ReLU())
        self.conv1 = nn.Sequential(self._make_pred_layer(Classifier_Module, [1, 6, 8, 12], [1, 6, 8, 12], 64),
                                   nn.BatchNorm2d(64), nn.ReLU())
        self.conv2 = nn.Sequential(self._make_pred_layer(Classifier_Module, [1, 6, 8, 12], [1, 6, 8, 12], 64),
                                   nn.BatchNorm2d(64), nn.ReLU())
        self.conv3 = nn.Sequential(self._make_pred_layer(Classifier_Module, [1, 6, 8, 12], [1, 6, 8, 12], 64),
                                   nn.BatchNorm2d(64), nn.ReLU())
        self.conv4 = nn.Sequential(nn.Conv2d(64, 64, 3, 1, 1, 1), nn.BatchNorm2d(64), nn.ReLU())

        # 卷积块用于特征细化
        self.cbr1 = nn.Sequential(nn.Conv2d(64, 64, 3, 1, 1, 1), nn.BatchNorm2d(64), nn.ReLU())
        self.cbr2 = nn.Sequential(nn.Conv2d(64, 64, 3, 1, 1, 1), nn.BatchNorm2d(64), nn.ReLU())
        self.cbr3 = nn.Sequential(nn.Conv2d(64, 64, 3, 1, 1, 1), nn.BatchNorm2d(64), nn.ReLU())
        self.cbr4 = nn.Sequential(nn.Conv2d(64, 64, 3, 1, 1, 1), nn.BatchNorm2d(64), nn.ReLU())

        # 注意力模块
        self.Att0 = Att()
        self.Att1 = Att()
        self.Att2 = Att()
        self.Att3 = Att()
        self.Att4 = Att()

        # 最终预测层
        self.map = nn.Conv2d(64, 1, 7, 1, 3)

        # Focus模块用于特征融合和细化
        self.fm1 = Focus(64, 64)
        self.fm2 = Focus(64, 64)
        self.fm3 = Focus(64, 64)
        self.fm4 = Focus(64, 64)

        self.out_map = nn.Conv2d(64, 1, 7, 1, 3)

        # 将ReLU设置为inplace操作以节省内存
        for m in self.modules():
            if isinstance(m, nn.ReLU):
                m.inplace = True

    def _make_pred_layer(self, block, dilation_series, padding_series, num_classes):
        return block(dilation_series, padding_series, num_classes)

    def forward(self, x):
        """
        前向传播函数
        Args:
            x: 输入图像 (B, 3, 704, 704)
        Returns:
            stage_loss1: 全局分支的预测结果列表 [(B, 1, 704, 704), ...]
            stage_loss2: 细化分支的预测结果列表 [(B, 1, 704, 704), ...]
        """
        # 通过PVT-v2 backbone提取多尺度特征
        pvt = self.backbone(x)  # 主干网络特征提取
        x1 = pvt[0]  # (B, 64, 176, 176)   - 1/4 scale
        x2 = pvt[1]  # (B, 128, 88, 88)    - 1/8 scale
        x3 = pvt[2]  # (B, 320, 44, 44)    - 1/16 scale
        x4 = pvt[3]  # (B, 512, 22, 22)    - 1/32 scale

        # 通过1x1卷积统一通道数到64
        x_4 = self.cr4(x4)  # (B, 64, 22, 22)
        x_3 = self.cr3(x3)  # (B, 64, 44, 44)
        x_2 = self.cr2(x2)  # (B, 64, 88, 88)
        x_1 = self.cr1(x1)  # (B, 64, 176, 176)

        stage_loss1 = list()  # 存储全局分支的预测
        stage_loss2 = list()  # 存储细化分支的预测
        FeedBack_feature = None  # 反馈特征初始化为None

        # 迭代优化过程，进行两次迭代
        for iter in range(4):
            # 第一次迭代使用原始特征，后续迭代加入反馈特征
            if FeedBack_feature is None:
                f_1 = x_1  # (B, 64, 176, 176)
            else:
                f_1 = x_1 + FeedBack_feature  # 特征融合

            f_2 = x_2  # (B, 64, 88, 88)
            f_3 = x_3  # (B, 64, 44, 44)
            f_4 = x_4  # (B, 64, 22, 22)

            # 全局特征提取分支 - 自顶向下路径
            # 第一阶段
            gf0 = self.conv0(f_1)  # (B, 64, 176, 176)
            gf0 = gf0 * self.Att0(gf0)  # 注意力加权
            gf0 = F.interpolate(gf0, size=x_2.size()[2:], mode='bilinear')  # (B, 64, 88, 88)

            # 第二阶段
            gf1 = self.conv1(gf0 + f_2)  # 特征融合并进行卷积 (B, 64, 88, 88)
            gf1 = gf1 * self.Att1(gf1)  # 注意力加权
            gf1 = F.interpolate(gf1, size=x_3.size()[2:], mode='bilinear')  # (B, 64, 44, 44)

            # 第三阶段
            gf2 = self.conv2(gf1 + f_3)  # (B, 64, 44, 44)
            gf2 = gf2 * self.Att2(gf2)
            gf2 = F.interpolate(gf2, size=x_4.size()[2:], mode='bilinear')  # (B, 64, 22, 22)

            # 第四阶段
            gf3 = self.conv3(gf2 + f_4)  # (B, 64, 22, 22)
            gf3 = gf3 * self.Att3(gf3)
            gf3 = self.conv4(gf3)  # 额外的特征处理
            gf_pre = self.map(gf3)  # (B, 1, 22, 22) - 全局预测图

            # 细化分支 - 自底向上路径
            # 第四层细化
            rf4 = f_4 + gf3  # (B, 64, 22, 22)
            rf4 = self.cbr1(rf4)
            rf4 = F.interpolate(rf4, size=x_3.size()[2:], mode='bilinear')  # (B, 64, 44, 44)

            # 第三层细化
            rf3 = rf4 + f_3  # (B, 64, 44, 44)
            rf3 = self.cbr2(rf3)
            rf3 = F.interpolate(rf3, size=x_2.size()[2:], mode='bilinear')  # (B, 64, 88, 88)

            # 第二层细化
            rf2 = rf3 + f_2  # (B, 64, 88, 88)
            rf2 = self.cbr3(rf2)
            rf2 = F.interpolate(rf2, size=x_1.size()[2:], mode='bilinear')  # (B, 64, 176, 176)

            # 第一层细化
            rf1 = rf2 + f_1  # (B, 64, 176, 176)
            rf1 = self.cbr4(rf1)
            rf1_map = self.out_map(rf1)  # (B, 1, 176, 176) - 细化预测图

            # 更新反馈特征
            FeedBack_feature = rf1  # 用于下一次迭代

            # 将预测图上采样到原始输入大小
            gf_pre = F.interpolate(gf_pre, size=x.size()[2:], mode='bilinear')  # (B, 1, 704, 704)
            rf1_map = F.interpolate(rf1_map, size=x.size()[2:], mode='bilinear')  # (B, 1, 704, 704)

            stage_loss1.append(gf_pre)  # 存储全局分支预测结果
            stage_loss2.append(rf1_map)  # 存储细化分支预测结果

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
