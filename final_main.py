import os
import torch
import numpy as np
import torch.nn as nn
from tqdm import tqdm   
import torch.nn.functional as F

from models.PConv_model import PConvUNet
from data_prep import Data_prep
from net import VGG16FeatureExtractor
from loss import InpaintingLoss, IOA_Loss
from torch.utils.data import TensorDataset, DataLoader, random_split
from tensorboardX import SummaryWriter
from torch.utils import data
import yaml

from util.io import load_ckpt
from util.io import save_ckpt
from torchvision.utils import make_grid
from torchvision.utils import save_image
import matplotlib.pyplot as plt


class InfiniteSampler(data.sampler.Sampler):
    def __init__(self, num_samples):
        self.num_samples = num_samples

    def __iter__(self):
        return iter(self.loop())

    def __len__(self):
        return 2 ** 31

    def loop(self):
        i = 0
        order = np.random.permutation(self.num_samples)
        while True:
            yield order[i]
            i += 1
            if i >= self.num_samples:
                np.random.seed()
                order = np.random.permutation(self.num_samples)
                i = 0


class Trainer:
    
    def __init__(self, config_path):
        self.config_path = config_path
        self.load_config(config_path)
        
        self.input_channels = self.config['model']['input_channels']
        self.batch_size = self.config['training']['batch_size']
        self.log_interval = self.config['training']['log_interval']
        self.max_iter = self.config['training']['max_iter']
        self.lr = self.config['training']['lr']
        
        self.img_size = self.config['model']['img_size']
        self.save_model_interval = self.config['training']['save_model_interval']
        self.vis_interval = self.config['training']['vis_interval']
        self.input_channels = self.config['model']['input_channels']
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
        self.input_channels = self.config['model']['input_channels']
        self.LAMBDA_DICT = self.config['model']['LAMBDA_DICT']
        
        self.data_path = self.config['paths']['data_path']
        self.save_dir = self.config['paths']['save_dir']
        
        self.writer = writer = SummaryWriter(log_dir=self.config['paths']['log_dir'])
        self.create_savedir(self.save_dir)
        
        self.resume = self.config['training']['resume']
    
    def load_config(self, config_file):
        with open(config_file, 'r') as ymlfile:
            self.config = yaml.safe_load(ymlfile)
        print(self.config)
    
    def create_savedir(self, save_dir):
        if not os.path.exists(save_dir):
            os.makedirs('{:s}/images'.format(save_dir))
            os.makedirs('{:s}/ckpt'.format(save_dir))
            
    def create_dataloader(self):
        print('Loading Data')
        numpy_array = np.load('/tng4/users/skayasth/Yearly/2024/Feb/Gated_2024/deepfillv2-pytorch/numpy_array.npy')
        print('Creating Dataloader')
        tensor_array = torch.tensor(numpy_array, dtype=torch.float32)
        total_size = len(tensor_array)
        train_size = int(0.8 * total_size)
        eval_size = total_size - train_size
            
        train_dataset, eval_dataset = random_split(tensor_array, [train_size, eval_size])
        
        train_dataloader = DataLoader(
        train_dataset, batch_size=self.batch_size,
        sampler=InfiniteSampler(len(train_dataset)),
        
        num_workers=10)

        # Creating the eval DataLoader
        eval_dataloader = DataLoader(
            eval_dataset, batch_size=self.batch_size,
            # No need for the InfiniteSampler for evaluation
            shuffle=False,
            num_workers=10)
        
        return train_dataloader, eval_dataloader
    
    def define_model(self):
        print('Creating Model')
        self.model = PConvUNet(self.input_channels).to(self.device)
        self.optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.model.parameters()), lr=self.lr)
        self.criterion = InpaintingLoss(VGG16FeatureExtractor()).to(self.device)
        # self.criterion_IOA = IOA_Loss().to(self.device)
        
        if self.resume:
            start_iter = load_ckpt(
                self.resume, [('model', self.model)], [('optimizer', self.optimizer)])
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = self.lr
            print('Starting from iter ', start_iter)

        
    def create_mask(self, batch_size, channels, height, width, seed=None):
        if seed is not None:
            torch.manual_seed(seed)
            # if torch.cuda.is_available():
            #     torch.cuda.manual_seed_all(seed)
                
        mask_first_channel = torch.rand(batch_size, 1, height, width) < 0.05  # 5% data present, 90% data missing in the first channel
        mask_other_channels = torch.ones(batch_size, channels - 1, height, width)  # All data present in other channels

        # Concatenate the masks for the first channel and the other channels
        mask_combined = torch.cat([mask_first_channel, mask_other_channels], dim=1).float()

        return mask_combined
    
    def save_img(self, real, mask, fake, iter):
        
        
        fake = fake[0, 0, :, :] * 322
        real = real[0, 0, :, :] * 322
        mask = mask[0, 0, :, :]
        
        # fake = fake[0, 0, :, :].detach().cpu().numpy()  * 322
        # real = real[0, 0, :, :].cpu().detach().numpy() * 322 
        # mask = mask[0, 0, :, :].cpu().detach().numpy()
        
        masked = real * mask
        
        grid1 = torch.cat((real, mask * 322), dim=0)    
        grid2 = torch.cat((masked, fake), dim=0)
        grid = torch.cat((grid1, grid2), dim=1)
        
        
        
        filename_t = os.path.join(self.save_dir, 'images', '{:d}.png'.format(iter))
        save_image(grid, filename_t)  
        
        
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

        
        plt.tight_layout()
        
        
        filename_n= os.path.join(self.save_dir, 'images', '{:d}_n.png'.format(iter))
        plt.savefig(filename_n)
        plt.clf()
        
    
    def train(self):
        start_iter = 0
        self.define_model()
        train_dataloader, eval_dataloader = self.create_dataloader()
        iterator_train = iter(train_dataloader)
        
        tqdm_iterator = tqdm(range(start_iter, self.max_iter), desc='Training')
        for i in tqdm_iterator:
            self.model.train()
            image = next(iterator_train)
            gt = image
            
            
            batch, channels, height, width = image.shape
            mask = self.create_mask(self.batch_size, channels, height, width)
           
            image = image.to(self.device)
            mask = mask.to(self.device)
            gt = gt.to(self.device)
            
            output = self.model(image, mask)
            loss_dict = self.criterion(image, mask, output, gt)
            
            self.loss = 0.0
            for key, coef in self.LAMBDA_DICT.items():
                value = coef * loss_dict[key]
                self.loss += value
                if (i + 1) % self.log_interval == 0:
                
                    tqdm_iterator.set_postfix(loss=f'{self.loss.item():.2f}', refresh=True)
                    self.writer.add_scalar('loss_{:s}'.format(key), value.item(), i + 1)
            
            # self.loss = self.criteria_IOA(output, gt)
            
            if (i + 1) % self.save_model_interval == 0 or (i + 1) == self.max_iter:
                save_ckpt('{:s}/ckpt/{:d}.pth'.format(self.save_dir, i + 1),
                        [('model', self.model)], [('optimizer', self.optimizer)], i + 1)
            
            # print("loss = {:.2f}".format(self.loss.item()))
            
            tqdm_iterator.set_postfix(loss=f'{self.loss.item():.4f}', refresh=True)
            self.optimizer.zero_grad()
            self.loss.backward()
            self.optimizer.step()
            
            
            if i%self.vis_interval == 0:
                self.model.eval()
                eval_loss = 0.0
                eval_steps = 0
                with torch.no_grad():  # Disable gradient computation
                    for j,eval_image in enumerate(eval_dataloader):  # Assuming eval_image is directly usable
                        if len(eval_image) != self.batch_size:
                            continue
                        eval_image = eval_image.to(self.device)
                        eval_gt = eval_image
                        eval_mask = self.create_mask(self.batch_size, channels, height, width, seed=i).to(self.device)
                        # If you have masks and ground truths in your eval dataset, unpack them here
                        
                        eval_output = self.model(eval_image, eval_mask)  # Forward pass
                        
                        eval_loss_dict = self.criterion(eval_image, eval_mask, eval_output, eval_gt)  # Compute loss
                        
                        for key, coef in self.LAMBDA_DICT.items():
                            eval_loss += coef * eval_loss_dict[key].item()
                        eval_steps += 1
                        
                        
                        
                avg_eval_loss = eval_loss / eval_steps
                
                print(f"Average eval loss: {avg_eval_loss:.4f}")
                
                
                self.save_img(image, mask, output, i)
            
           
            

            
   

            
        pass
    
if __name__ == "__main__":
    config_path = 'config.yaml'
    trainer = Trainer(config_path)
    trainer.train()
