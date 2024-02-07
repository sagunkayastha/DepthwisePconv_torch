import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class PConv2D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding='valid', dilation=1, bias=True, init_method='kaiming'):
        super(PConv2D, self).__init__()
        
        padding = 'valid'
        self.in_channels = in_channels
        self.data_format = 'channels_first'
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.out_channels = out_channels
        
        
        # Initialize depthwise convolutions for both input and mask
        
        self.input_conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, bias=bias)
        self.mask_conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, bias=False)
        self.initialize_weights(self.input_conv.weight,method='kaiming')
        nn.init.constant_(self.mask_conv.weight, 1.0)
        
        # mask is not updated
        for param in self.mask_conv.parameters():
            param.requires_grad = False
        
        # Assuming kernel_size is a tuple
        kernel_size = (kernel_size, kernel_size)
        self.kernel_size = kernel_size
        self.window_size = kernel_size[0] * kernel_size[1]

        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
            self.bias.data.fill_(0)  # Initialize bias to zero
        else:
            self.register_parameter('bias', None)

    
    def initialize_weights(self, m_weights, method='kaiming'):
        
        if method == 'gaussian':
            nn.init.normal_(m_weights, 0.0, 0.02)
        
        elif method == 'kaiming':
            nn.init.kaiming_normal_(m_weights, mode='fan_out', nonlinearity='relu')
        elif method == 'xavier':
            nn.init.xavier_normal_(m_weights, gain=math.sqrt(2))
        elif method == 'orthogonal':
            nn.init.orthogonal_(m_weights, gain=math.sqrt(2))
        elif method == 'default':
            pass
        elif method == 'mask':
            
            nn.init.constant_(m_weights, 1.0)
        else:
            raise ValueError("Unsupported initialization method")

        if self.input_conv.bias is not None:
            nn.init.constant_(self.input_conv.bias, 0)          
            
    
    
    def forward(self, inputs):
        if not isinstance(inputs, list) or len(inputs) != 2:
            raise ValueError('Input must be a list of two tensors [img, mask]')
        # self.out_channels = self.compute_output_shape(inputs[0].shape)
        # print(self.out_channels)
        image, mask = inputs

        # Apply padding manually for 'same' padding
        pad_height = max(self.kernel_size[0] - 1, 0)
        pad_width = max(self.kernel_size[1] - 1, 0)
        padding = (pad_width // 2, pad_width - pad_width // 2, pad_height // 2, pad_height - pad_height // 2)
        image_padded = F.pad(image, padding, 'constant', 0)
        mask_padded = F.pad(mask, padding, 'constant', 0)
        
        # Perform depthwise convolution on both the image and the mask
        img_output = self.input_conv(image_padded * mask_padded)
        mask_output = self.mask_conv(mask_padded)
        
        # Calculate the mask ratio
        mask_ratio = self.window_size / (mask_output + 1e-8)
        mask_output = torch.clamp(mask_output, 0, 1)  # Clip values to be between 0 and 1
        mask_ratio = mask_ratio * mask_output

        # Normalize image output
        img_output = img_output * mask_ratio

        # Correct bias addition
        if self.bias is not None:
            # Ensure bias is correctly expanded to match img_output's dimensions
            expanded_bias = self.bias.view(1, self.out_channels, 1, 1).expand_as(img_output)
            img_output += expanded_bias

        return img_output, mask_output
    


