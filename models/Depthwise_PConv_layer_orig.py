import torch
import torch.nn as nn
import torch.nn.functional as F

def weights_init(init_type='gaussian'):
    def init_fun(m):
        classname = m.__class__.__name__
        if (classname.find('Conv') == 0 or classname.find(
                'Linear') == 0) and hasattr(m, 'weight'):
            if init_type == 'gaussian':
                nn.init.normal_(m.weight, 0.0, 0.02)
            elif init_type == 'xavier':
                nn.init.xavier_normal_(m.weight, gain=math.sqrt(2))
            elif init_type == 'kaiming':
                nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                nn.init.orthogonal_(m.weight, gain=math.sqrt(2))
            elif init_type == 'default':
                pass
            else:
                assert 0, "Unsupported initialization: {}".format(init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias, 0.0)

    return init_fun



class DepthwiseConv2D(nn.Module):
    def __init__(self, in_channels, kernel_size, stride=1, padding=0, dilation=1, bias=True, depth_multiplier=1):
        super(DepthwiseConv2D, self).__init__()
        self.out_channels = in_channels * depth_multiplier  # Adjusted for depth_multiplier
        self.depthwise = nn.Conv2d(
            in_channels=in_channels,
            out_channels=self.out_channels,  # Adjusted for depth_multiplier
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=in_channels,  # Ensures depthwise convolution
            bias=bias
        )

    def forward(self, x):
        x = self.depthwise(x)
        return x

    def initialize_weights(self, method='kaiming'):
        if method == 'kaiming':
            nn.init.kaiming_normal_(self.depthwise.weight, mode='fan_out', nonlinearity='relu')
        elif method == 'xavier':
            nn.init.xavier_normal_(self.depthwise.weight)
        else:
            raise ValueError("Unsupported initialization method")

        if self.depthwise.bias is not None:
            nn.init.constant_(self.depthwise.bias, 0)
            
            
class PConv2D_depthwise(nn.Module):
    def __init__(self, in_channels, kernel_size, depth_multiplier=4, stride=1, padding='valid', dilation=1, bias=True):
        super(PConv2D_depthwise, self).__init__()
        
        self.depth_multiplier = depth_multiplier 
        self.in_channels = in_channels
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.out_channels = in_channels * self.depth_multiplier
        padding = 'valid'
        
        # Initialize depthwise convolutions for both input and mask
        self.input_conv = DepthwiseConv2D(in_channels, kernel_size, stride, padding, dilation, bias=bias, depth_multiplier=self.depth_multiplier)
        self.mask_conv = DepthwiseConv2D(in_channels, kernel_size, stride, padding, dilation, bias=False, depth_multiplier=self.depth_multiplier)
        torch.nn.init.constant_(self.mask_conv.weight, 1.0)

        # mask is not updated
        for param in self.mask_conv.parameters():
            param.requires_grad = False
            
        # Assuming kernel_size is a tuple
        kernel_size = (kernel_size, kernel_size)
        self.kernel_size = kernel_size
        self.window_size = kernel_size[0] * kernel_size[1]

        if bias:
            self.bias = nn.Parameter(torch.Tensor(in_channels*self.depth_multiplier))
            self.bias.data.fill_(0)  # Initialize bias to zero
        else:
            self.register_parameter('bias', None)

    def forward(self, inputs):
        if not isinstance(inputs, list) or len(inputs) != 2:
            raise ValueError('Input must be a list of two tensors [img, mask]')

        image, mask = inputs
        
        # Apply padding manually for 'same' padding
        pad_height = max(self.kernel_size[0] - 1, 0)
        pad_width = max(self.kernel_size[1] - 1, 0)
        padding = (pad_width // 2, pad_width - pad_width // 2, pad_height // 2, pad_height - pad_height // 2)
        image_padded = F.pad(image, padding, 'constant', 0)
        mask_padded = F.pad(mask, padding, 'constant', 0)
        
        # Perform depthwise convolution on both the image and the mask
        img_output = self.input_conv(image_padded * mask_padded)
        with torch.no_grad():
            mask_output = self.mask_conv(mask_padded)
        print(img_output.sum())
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

# Utility function to calculate 'same' padding
def get_same_padding(kernel_size):
    pad_val = (kernel_size - 1) // 2
    return pad_val
