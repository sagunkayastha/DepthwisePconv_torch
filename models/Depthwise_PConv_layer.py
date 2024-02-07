import torch
import torch.nn as nn
import torch.nn.functional as F




class DepthwiseConv2D(nn.Module):
    def __init__(self, in_channels, kernel_size, stride=1, padding=0, dilation=1, bias=True, depth_multiplier=1, init_method='kaiming'):
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
        self.weight = self.depthwise.weight
        self.bias = self.depthwise.bias
    
    def forward(self, x):
        x = self.depthwise(x)
        return x           
            
            

class DepthwiseConv2D_mod(nn.Module):
    def __init__(self, in_channels, kernel_size, stride=1, padding=0, dilation=1, bias=True, depth_multiplier=1, mask_c=False):
        super(DepthwiseConv2D, self).__init__()
        self.out_channels = in_channels * depth_multiplier  # Adjusted for depth_multiplier
        self.depthwise = nn.Conv2d(
            in_channels=in_channels,
            out_channels=self.out_channels,  # Adjusted for depth multiplier
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=in_channels,  # Ensures depthwise convolution
            bias=bias
        )
        self.kernel_size = [kernel_size, kernel_size]
        # Initialize kernel mask
        

    def forward(self, x):      
        
        x = F.conv2d(x, self.kernel_weights, self.depthwise.bias, self.depthwise.stride, 
                     self.depthwise.padding, self.depthwise.dilation, self.depthwise.groups)
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
        
        self.initialize_weights(self.input_conv.weight,method='kaiming')
        nn.init.constant_(self.mask_conv.weight, 1.0)
            
        # Assuming kernel_size is a tuple
        kernel_size = (kernel_size, kernel_size)
        self.kernel_size = kernel_size
        self.window_size = kernel_size[0] * kernel_size[1]

        if bias:
            self.bias = nn.Parameter(torch.Tensor(in_channels*self.depth_multiplier))
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
        
        # if self.bias is not None:
        #     output_bias = self.bias.view(1, self.out_channels, 1, 1).expand_as(output)

        # # mask_sum is the sum of the binary mask at every partial convolution location
        # mask_is_zero = (output_mask == 0)
        # # temporarily sets zero values to one to ease output calculation 
        # mask_sum = output_mask.masked_fill_(mask_is_zero, 1.0)

        # # output at each location as follows:
        # # output = (W^T dot (X .* M) + b - b) / M_sum + b ; if M_sum > 0
        # # output = 0 ; if M_sum == 0
        # output = (output - output_bias) / mask_sum + output_bias
        # output = output.masked_fill_(mask_is_zero, 0.0)

        # # mask is updated at each location
        # new_mask = torch.ones_like(output)
        # new_mask = new_mask.masked_fill_(mask_is_zero, 0.0)
        
       
        return img_output, mask_output

    


# Utility function to calculate 'same' padding
def get_same_padding(kernel_size):
    pad_val = (kernel_size - 1) // 2
    return pad_val
