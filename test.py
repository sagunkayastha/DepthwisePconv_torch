import argparse
import torch
import os
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from torchvision.utils import save_image
from tqdm import tqdm
import matplotlib.pyplot as plt


from models.PConv_model import PConvUNet
from util.io import load_ckpt
import yaml


class Test:
    
    
    def __init__(self, config_path):
        self.config_path = config_path
        self.load_config(config_path)
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.save_dir = self.config['paths']['save_dir']
        self.ckpt = os.path.join(self.save_dir, 'ckpt', '99000.pth')
        self.input_channels = self.config['model']['input_channels']
        self.batch_size = 1
        self.load_model()
        
    def load_config(self, config_file):
        with open(config_file, 'r') as ymlfile:
            self.config = yaml.safe_load(ymlfile)
        print(self.config)
        
    def load_model(self):
        print('Creating Model')
        self.model = PConvUNet(self.input_channels).to(self.device)
        
        if self.ckpt:
            print('Loading Checkpoint')
            load_ckpt(self.ckpt, [('model', self.model)])
            
            
    def load_data(self):
        
        numpy_array = np.load('/tng4/users/skayasth/Yearly/2024/Feb/Gated_2024/generative-inpainting-pytorch/test_2019.npy')
        # numpy_array = np.load('/tng4/users/skayasth/Yearly/2024/Feb/Gated_2024/generative-inpainting-pytorch/numpy_array.npy')
        print('Creating Dataloader')
        tensor_array = torch.tensor(numpy_array, dtype=torch.float32)
        dataset = TensorDataset(tensor_array)
        test_dataloader = DataLoader(
            dataset, batch_size=self.batch_size,
            num_workers=0)
        return test_dataloader
    
    
    def create_mask(self, batch_size, channels, height, width):
        mask_first_channel = torch.rand(batch_size, 1, height, width) < 0.1  # 10% data present, 90% data missing in the first channel
        mask_other_channels = torch.ones(batch_size, channels - 1, height, width)  # All data present in other channels

        # Concatenate the masks for the first channel and the other channels
        mask_combined = torch.cat([mask_first_channel, mask_other_channels], dim=1).float()

        return mask_combined
    
    def save_img(self, real, mask, fake, iter):
        
        
        fake = fake[0, 0, :, :] * 322
        real = real[0, 0, :, :] * 322
        mask = mask[0, 0, :, :]
        masked = real * mask
        fake, real, mask, masked = fake.cpu().detach().numpy(), real.cpu().detach().numpy(), mask.cpu().detach().numpy(), masked.cpu().detach().numpy()
        
        fig, ax = plt.subplots(2,2, figsize=(40,20))
       # Plotting 'real'
        real_display = ax[0, 0].imshow(real)
        ax[0, 0].set_title('Real')
        ax[0, 0].axis('off')
        fig.colorbar(real_display, ax=ax[0, 0], fraction=0.046, pad=0.04)

        # Plotting 'mask'
        mask_display = ax[0, 1].imshow(mask, cmap='gray')
        ax[0, 1].set_title('Mask')
        ax[0, 1].axis('off')
        fig.colorbar(mask_display, ax=ax[0, 1], fraction=0.046, pad=0.04)

        # Plotting 'masked'
        masked_display = ax[1, 0].imshow(masked)
        ax[1, 0].set_title('Masked')
        ax[1, 0].axis('off')
        fig.colorbar(masked_display, ax=ax[1, 0], fraction=0.046, pad=0.04)

        # Plotting 'fake'
        fake_display = ax[1, 1].imshow(fake)
        ax[1, 1].set_title('Fake')
        ax[1, 1].axis('off')
        fig.colorbar(fake_display, ax=ax[1, 1], fraction=0.046, pad=0.04)

        
        real_expanded = np.expand_dims(real, axis=0)
        mask_expanded = np.expand_dims(mask, axis=0)
        masked_expanded = np.expand_dims(masked, axis=0)
        fake_expanded = np.expand_dims(fake, axis=0)

        # Combine them into a single NumPy array
        arr_to_save = np.array([real_expanded, mask_expanded, masked_expanded, fake_expanded])
        plt.tight_layout()
        
        filename_n= os.path.join('test_output', '{:d}_n.png'.format(iter))
        plt.savefig(filename_n)
        np.save(os.path.join('test_output', '{:d}.npy'.format(iter)), arr_to_save)
        plt.clf()
    
    def eval(self):
        test_dataloader = self.load_data()
        iterator_train = iter(test_dataloader)
        
        
        self.model.eval()
        
        image = [x for x in next(iterator_train)][0]
        gt = image
        batch, channels, height, width = image.shape
        mask = self.create_mask(self.batch_size, channels, height, width)
        
        image = image.to(self.device)
        mask = mask.to(self.device)
        gt = gt.to(self.device)
        
        output = self.model(image, mask)
       
         
            
        # output_comp = mask[0, 0, :, :] * image[0, 0, :, :] + (1 - mask[0, 0, :, :]) * output[0, 0, :, :]
        
        self.save_img(image, mask, output, 0)
        
        
if __name__ == "__main__":
    config_path = 'config.yaml'
    test = Test(config_path)
    test.eval()
    print("Done!")