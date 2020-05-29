import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

# Encoder
class mobilnet(nn.Module):
    def __init__(self,pretrained=True, num_class=21):
        super().__init__()
        self.features = models.mobilenet_v2(pretrained=pretrained).features
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.drop=nn.Dropout()
        self.fc = nn.Linear(1280, num_class)

    def forward(self, input):
        x = self.features[0](input)

        x = self.features[1](x)
        x = self.features[2](x)
        x2 = self.features[3](x)        #56
        x = self.features[4](x2)
        x = self.features[5](x)
        x3 = self.features[6](x)        #28
        x = self.features[7](x3)
        x = self.features[8](x)
        x = self.features[9](x)
        x = self.features[10](x)
        x = self.features[11](x)
        x = self.features[12](x)
        x4 = self.features[13](x)
        x = self.features[14](x4)
        x = self.features[15](x)
        x = self.features[16](x)
        x = self.features[17](x)
        x5 = self.features[18](x)       #7

        pred = self.avgpool(x5)
        pred = torch.flatten(pred, 1)
        pred = self.drop(pred)
        pred = self.fc(pred)

        return x2,x3,x4,x5,pred

# Dual Attention Block
class ChannelAttentionModule(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.cam=nn.Sequential(
            nn.AdaptiveAvgPool2d(output_size=(1, 1)),
            nn.Conv2d(in_channels, in_channels, kernel_size=1),
            nn.PReLU(num_parameters=in_channels),
            nn.Conv2d(in_channels, in_channels, kernel_size=1),
            nn.Sigmoid()
        )

        self.sam=nn.Sequential(
            nn.Conv2d(in_channels, in_channels,kernel_size=3,padding=1,groups=in_channels),
            nn.Sigmoid()
        )

        self.conv=nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.PReLU(num_parameters=in_channels),
            nn.Conv2d(in_channels, out_channels,kernel_size=3,padding=1),
            nn.BatchNorm2d(out_channels),
            nn.PReLU(num_parameters=out_channels),
        )

        self.t = torch.nn.Parameter(torch.Tensor([1.00]))

        self.init_weight()

    def init_weight(self):
        for layer in self.conv:
            if isinstance(layer, nn.Conv2d):
                torch.nn.init.xavier_uniform_(layer.weight)
        for layer in self.cam:
            if isinstance(layer, nn.Conv2d):
                torch.nn.init.xavier_uniform_(layer.weight)
        for layer in self.sam:
            if isinstance(layer, nn.Conv2d):
                torch.nn.init.xavier_uniform_(layer.weight)

    def forward(self, input):
        x_c=self.cam(input)
        out = torch.mul(input, x_c)

        x_s=self.sam(input)
        out=torch.mul(out,x_s)

        out=out*self.t

        out=torch.add(out,input)
        out=self.conv(out)

        return out

# Feature Fusion Block
class FeatureFusionModule(torch.nn.Module):
    def __init__(self,channel_1, channel_2, Up_2=False, Up_1=False):
        super().__init__()
        self.up1=Up_1
        self.up2=Up_2

        if self.up1==True:
            self.dconv1 = nn.ConvTranspose2d(in_channels=channel_1, out_channels=channel_1, kernel_size=4, stride=2, padding=1, bias=False)

        if self.up2 == True:
            self.dconv2 = nn.ConvTranspose2d(in_channels=channel_2, out_channels=channel_2, kernel_size=4, stride=2, padding=1, bias=False)

        self.conv1=nn.Sequential(
            nn.Conv2d(channel_1+channel_2, channel_1+channel_2, kernel_size=3, padding=1, groups=channel_1+channel_2),
            nn.Conv2d(channel_1+channel_2, channel_1, kernel_size=1, padding=0),
            nn.BatchNorm2d(channel_1),
            nn.PReLU(num_parameters=channel_1),
            nn.Conv2d(channel_1, channel_1, kernel_size=3, padding=1,groups=channel_1),
            nn.Conv2d(channel_1, channel_1, kernel_size=1, padding=0),
            nn.BatchNorm2d(channel_1),
            nn.PReLU(num_parameters=channel_1),
        )

        # DAB
        self.conv2=ChannelAttentionModule(channel_1, channel_1)

        for layer in self.conv1:
            if isinstance(layer, nn.Conv2d):
                torch.nn.init.xavier_uniform_(layer.weight)

    def forward(self, input_1, input_2):
        if self.up2==True:
            input_2 = self.dconv2(input_2)
        if self.up1==True:
            input_1 = self.dconv1(input_1)

        input=torch.cat((input_1,input_2),dim=1)

        x=self.conv1(input)
        x=self.conv2(x)

        out=x+input_1
        
        return out

# ASPP
class ASPP(nn.Module):
    def __init__(self, in_channel=512, depth=256):
        super(ASPP, self).__init__()
        # global average pooling : init nn.AdaptiveAvgPool2d ;also forward torch.mean(,,keep_dim=True)
        self.mean = nn.AdaptiveAvgPool2d((1, 1))
        self.conv = nn.Conv2d(in_channel, depth, 1, 1)
        self.atrous_block1 = nn.Conv2d(in_channel, depth, 1, 1)
        self.atrous_block6 = nn.Conv2d(in_channel, depth, 3, 1, padding=6, dilation=6)
        self.atrous_block12 = nn.Conv2d(in_channel, depth, 3, 1, padding=12, dilation=12)
        self.atrous_block18 = nn.Conv2d(in_channel, depth, 3, 1, padding=18, dilation=18)

        self.conv_1x1_output = nn.Conv2d(depth * 5, depth * 2, 1, 1)

        torch.nn.init.xavier_uniform_(self.conv.weight)
        torch.nn.init.xavier_uniform_(self.atrous_block1.weight)
        torch.nn.init.xavier_uniform_(self.atrous_block6.weight)
        torch.nn.init.xavier_uniform_(self.atrous_block12.weight)
        torch.nn.init.xavier_uniform_(self.atrous_block18.weight)
        torch.nn.init.xavier_uniform_(self.conv_1x1_output.weight)

    def forward(self, x):
        size = x.shape[2:]

        image_features = self.mean(x)
        image_features = self.conv(image_features)
        image_features = F.upsample(image_features, size=size, mode='bilinear')

        atrous_block1 = self.atrous_block1(x)

        atrous_block6 = self.atrous_block6(x)

        atrous_block12 = self.atrous_block12(x)

        atrous_block18 = self.atrous_block18(x)

        net = self.conv_1x1_output(torch.cat([image_features, atrous_block1, atrous_block6,
                                              atrous_block12, atrous_block18], dim=1))
        return net

# Multi-Scales Dilation Convolution Block
class MCB(nn.Module):
    def __init__(self,in_channel, out_channel):
        super().__init__()
        self.f_loc=nn.Sequential(
            nn.Conv2d(in_channel,in_channel,kernel_size=3,padding=1,groups=in_channel),
            nn.Conv2d(in_channel,out_channel,kernel_size=1,padding=0,groups=1)
        )

        self.f_sur=nn.Sequential(
            nn.Conv2d(in_channel,in_channel,kernel_size=3,padding=2,groups=in_channel,dilation=2),
            nn.Conv2d(in_channel,out_channel,kernel_size=1,padding=0,groups=1)
        )

        self.conv=nn.Sequential(
            nn.BatchNorm2d(int(out_channel*2),eps=1e-3),
            nn.PReLU(int(out_channel*2)),
            nn.Conv2d(int(out_channel*2),out_channel, kernel_size=1),
        )

        for layer in self.conv:
            if isinstance(layer, nn.Conv2d):
                torch.nn.init.xavier_uniform_(layer.weight)

    def forward(self, input):
        loc=self.f_loc(input)
        sur=self.f_sur(input)

        joi=torch.cat([loc,sur],1)
        joi=self.conv(joi)

        return joi

# Classification Assisted Segmentation Block
class CAS(nn.Module):
    def __init__(self,in_channel, num_class=21 ):
        super().__init__()
        self.pre_conv=nn.Sequential(
            nn.Conv2d(in_channel, in_channel, kernel_size=3, padding=1,groups=in_channel),
            nn.Conv2d(in_channel, num_class, kernel_size=1, padding=0, groups=1),
            nn.BatchNorm2d(num_class),
            nn.Sigmoid(),
        )

        self.fc1 = nn.Linear(num_class, num_class)
        self.fc2 = nn.Linear(num_class, 1)

        self.conv=nn.Sequential(
            nn.BatchNorm2d(num_class),
            nn.PReLU(num_class),
            nn.Conv2d(num_class,num_class,kernel_size=3,padding=1)
        )

        for layer in self.pre_conv:
            if isinstance(layer, nn.Conv2d):
                torch.nn.init.xavier_uniform_(layer.weight)
        for layer in self.conv:
            if isinstance(layer, nn.Conv2d):
                torch.nn.init.xavier_uniform_(layer.weight)

    def forward(self, c_out, s_out):
        aux_weight=self.fc1(c_out)
        alpha=self.fc2(aux_weight)

        aux_weight=aux_weight.view(c_out.shape[0],c_out.shape[1],1,1)
        alpha=alpha.view(c_out.shape[0],1,1,1)

        input=self.pre_conv(s_out)
        residual=input

        x=torch.mul(aux_weight,input)
        x=x*alpha

        x=x+residual
        x=self.conv(x)

        return x

# Semantic Embedding Block
class SEB(nn.Module):
    def __init__(self, in_channel_1,in_channel_2):
        super().__init__()
        self.conv=nn.Sequential(
            nn.Conv2d(in_channel_2, in_channel_1, kernel_size=3,padding=1),
            nn.BatchNorm2d(in_channel_1),
            nn.PReLU(in_channel_1),
        )

        for layer in self.conv:
            if isinstance(layer, nn.Conv2d):
                torch.nn.init.xavier_uniform_(layer.weight)

    def forward(self,in_low,in_high):
        hi = self.conv(in_high)
        hi = nn.functional.interpolate(hi, scale_factor=2, mode='bilinear')
        out = torch.mul(in_low,hi)
        return out

# EEDNet
class EEDNet(nn.Module):
    def __init__(self,num_classes=19):
        super(EEDNet, self).__init__()
        self.down = mobilnet(pretrained=True, num_class=num_classes)

        channel_vec = [24, 32, 96, 1280]
        channel_vec_out = [24, 32, 64, 128]

        self.seb2 = SEB(in_channel_1=channel_vec[0],in_channel_2=channel_vec[1])
        self.seb3 = SEB(in_channel_1=channel_vec[1],in_channel_2=channel_vec[2])
        self.seb4 = SEB(in_channel_1=channel_vec[2],in_channel_2=channel_vec[3])

        self.conv2 = MCB(int(channel_vec_out[0]), int(channel_vec_out[0]))
        self.conv3 = MCB(int(channel_vec_out[1]), int(channel_vec_out[1]))
        self.conv4 = MCB(int(channel_vec_out[2]), int(channel_vec_out[2]))
        self.conv5 = MCB(int(channel_vec[3]), int(channel_vec_out[3]))

        self.aspp = ASPP(in_channel=int(channel_vec_out[3]), depth=int(channel_vec_out[3]/2))

        self.cam2 = ChannelAttentionModule(int(channel_vec[0]), int(channel_vec_out[0]))
        self.cam3 = ChannelAttentionModule(int(channel_vec[1]), int(channel_vec_out[1]))
        self.cam4 = ChannelAttentionModule(int(channel_vec[2]), int(channel_vec_out[2]))

        self.up45 = FeatureFusionModule(int(channel_vec_out[2]), int(channel_vec_out[3]), Up_2=True, Up_1=False)
        self.up34 = FeatureFusionModule(int(channel_vec_out[1]), int(channel_vec_out[2]), Up_2=True, Up_1=False)
        self.up23 = FeatureFusionModule(int(channel_vec_out[0]), int(channel_vec_out[1]), Up_2=True, Up_1=False)

        self.cgs=CAS(int(channel_vec_out[0]), num_classes)

    def forward(self, input):
        x2,x3,x4,x5,aux_loss_s5=self.down(input)

        x2 = self.seb2(x2, x3)
        x3 = self.seb3(x3, x4)
        x4 = self.seb4(x4, x5)

        x4 = self.cam4(x4)
        x3 = self.cam3(x3)
        x2 = self.cam2(x2)

        x5 = self.conv5(x5)
        x5 = self.aspp(x5)

        x4 = self.conv4(x4)
        x3 = self.conv3(x3)
        x2 = self.conv2(x2)

        x4 = self.up45(x4, x5)
        x3 = self.up34(x3, x4)
        x2 = self.up23(x2, x3)

        res = self.cgs(aux_loss_s5, x2)

        res = nn.functional.interpolate(res, scale_factor=4, mode='bilinear')

        return res

if __name__ == "__main__":
    model = EEDNet(num_classes=19)
    print(model)











