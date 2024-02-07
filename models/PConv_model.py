import torch
import torch.nn as nn
import torch.nn.functional as F

from models.PConv_layer import PConv2D
from models.Depthwise_PConv_layer import PConv2D_depthwise


class EncoderLayer(nn.Module):
    def __init__(self, in_channels, kernel_size, depth_multiplier, stride=2, bn=True):
        super().__init__()
        
        self.pconv = PConv2D_depthwise(in_channels,  kernel_size, depth_multiplier,stride=stride, padding='same')
        self.bn = nn.BatchNorm2d(in_channels * depth_multiplier) if bn else None
        self.activation = nn.ReLU()
        # self.leaky_relu = nn.LeakyReLU(0.2)

    def forward(self, img_in, mask_in):
        img_out, mask_out = self.pconv([img_in, mask_in])
        if self.bn is not None:
            img_out = self.bn(img_out)
        img_out = self.activation(img_out)
        return img_out, mask_out
    
class DecoderLayer(nn.Module):
    def __init__(self,in_channels, out_channels, kernel_size, bn=True):
        super().__init__()
        # self.upsampling_mode = upsampling_mode
        
        self.up = nn.Upsample(scale_factor=2, mode='nearest')
        
        self.kernel_size = kernel_size
        self.pconv = PConv2D(in_channels=in_channels+out_channels, out_channels=out_channels,  kernel_size=kernel_size, padding='valid', stride=1, dilation=1)
        
        self.bn = nn.BatchNorm2d(out_channels) if bn else None
        self.leaky_relu = nn.LeakyReLU(0.4)

    def forward(self, img_in, mask_in, e_conv, e_mask):
        
        img_in_up = self.up(img_in)
        mask_in_up = self.up(mask_in)
        
        
        concat_img = torch.cat([e_conv, img_in_up], dim=1)
        concat_mask = torch.cat([e_mask, mask_in_up], dim=1)
        img_out, mask_out = self.pconv([concat_img, concat_mask])
        
        if self.bn is not None:
            img_out = self.bn(img_out)
        img_out = self.leaky_relu(img_out)
        return img_out, mask_out

class PConvUNet(nn.Module):
    def __init__(self, img_channels=3):
        super().__init__()
        # Define encoder layers
        self.e1 = EncoderLayer(img_channels, kernel_size=7, depth_multiplier=4, bn=False)
        self.e2 = EncoderLayer(img_channels * 4, kernel_size=5, depth_multiplier=2)
        self.e3 = EncoderLayer(img_channels * 8, kernel_size=5, depth_multiplier=2)
        self.e4 = EncoderLayer(img_channels * 16, kernel_size=3, depth_multiplier=2)
        self.e5 = EncoderLayer(img_channels * 32, kernel_size=3, depth_multiplier=2)
        self.e6 = EncoderLayer(img_channels * 64, kernel_size=3, depth_multiplier=1)
        # self.e7 = EncoderLayer(img_channels * 128, kernel_size=3, depth_multiplier=1)
        
        bottle_neck = img_channels * 64
        
        # self.d1 = DecoderLayer(bottle_neck, bottle_neck, kernel_size=3)

        
        self.d2 = DecoderLayer(bottle_neck, bottle_neck, kernel_size=3)
        self.d3 = DecoderLayer(bottle_neck, bottle_neck//2, kernel_size=3)
        self.d4 = DecoderLayer(bottle_neck//2, bottle_neck//4, kernel_size=3)
        self.d5 = DecoderLayer(bottle_neck//4, bottle_neck//8, kernel_size=3)
        self.d6 = DecoderLayer(bottle_neck//8, bottle_neck//16, kernel_size=3)
        self.d7 = DecoderLayer(bottle_neck//16, img_channels, kernel_size=3)
        # Define final output layer
        self.final_conv = nn.Conv2d(img_channels, 1, kernel_size=1)
        self.final_activation = nn.Sigmoid()
        # self.final_activation = nn.ReLU()

    def forward(self, inputs_img, inputs_mask):
        # Forward pass through encoder layers
        e_conv1, e_mask1 = self.e1(inputs_img, inputs_mask)
        
        e_conv2, e_mask2 = self.e2(e_conv1, e_mask1)
        e_conv3, e_mask3 = self.e3(e_conv2, e_mask2)
        e_conv4, e_mask4 = self.e4(e_conv3, e_mask3)
        e_conv5, e_mask5 = self.e5(e_conv4, e_mask4)
        e_conv6, e_mask6 = self.e6(e_conv5, e_mask5)
        
        
        
        # # # Continue through all encoder and decoder layers...
        # e_conv7, e_mask7 = self.e7(e_conv6, e_mask6)
        # d_conv1, d_mask1 = self.d1(e_conv7, e_mask7, e_conv6, e_mask6)
        d_conv2, d_mask2 = self.d2(e_conv6, e_mask6, e_conv5, e_mask5)
        
        # d_conv2, d_mask2 = self.d2(d_conv1, d_mask1, e_conv5, e_mask5)
        d_conv3, d_mask3 = self.d3(d_conv2, d_mask2, e_conv4, e_mask4)
        d_conv4, d_mask4 = self.d4(d_conv3, d_mask3, e_conv3, e_mask3)
        d_conv5, d_mask5 = self.d5(d_conv4, d_mask4, e_conv2, e_mask2)
        d_conv6, d_mask6 = self.d6(d_conv5, d_mask5, e_conv1, e_mask1)
        
        d_conv7, d_mask7 = self.d7(d_conv6, d_mask6, inputs_img, inputs_mask)
        
        
        # # # Final output layer
        outputs = self.final_conv(d_conv7)
        outputs = self.final_activation(outputs)
        
        
        return outputs


