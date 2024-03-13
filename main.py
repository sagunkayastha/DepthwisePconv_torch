import torch
import numpy as np
import torch.nn as nn
from tqdm import tqdm   
import torch.nn.functional as F
import os
from models.PConv_model import PConvUNet
from data_prep import Data_prep
from net import VGG16FeatureExtractor
from loss import InpaintingLoss
from torch.utils.data import TensorDataset, DataLoader
from tensorboardX import SummaryWriter
data_path = '/tng4/users/skayasth/Yearly/2023/Jan/TCEQ/Data_for_PCNN'
from torch.utils import data


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


save_dir = 'snapshots/default'

if not os.path.exists(save_dir):
    os.makedirs('{:s}/images'.format(save_dir))
    os.makedirs('{:s}/ckpt'.format(save_dir))


numpy_array = np.load('/tng4/users/skayasth/Yearly/2024/Feb/Gated_2024/deepfillv2-pytorch/numpy_array.npy')
tensor_array = torch.tensor(numpy_array, dtype=torch.float32)


LAMBDA_DICT = {
    'valid': 1.0, 'hole': 6.0, 'tv': 0.1, 'prc': 0.05, 'style': 120.0}
writer = SummaryWriter(log_dir='logs')
batch_size = 16
log_interval = 100
max_iter = 1000000

dataset = TensorDataset(tensor_array)
train_dataloader = DataLoader(
    dataset, batch_size=batch_size,
    sampler=InfiniteSampler(len(dataset)),
    num_workers=0)

# train_dataloader = DataLoader(dataset, batch_size=batch_size,
#                                                 shuffle=True,
#                                                 drop_last=True,
#                                                 num_workers=0,
#                                                 pin_memory=True)

iterator_train = iter(train_dataloader)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = PConvUNet(5).to(device)

start_iter = 0
lr = 0.00001
save_model_interval = 1000

optimizer = torch.optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
criterion = InpaintingLoss(VGG16FeatureExtractor()).to(device)


def create_mask(batch_size, channels, height, width):
    mask_first_channel = torch.rand(batch_size, 1, height, width) < 0.1  # 10% data present, 90% data missing in the first channel
    mask_other_channels = torch.ones(batch_size, channels - 1, height, width)  # All data present in other channels

    # Concatenate the masks for the first channel and the other channels
    mask_combined = torch.cat([mask_first_channel, mask_other_channels], dim=1).float()

    return mask_combined

for i in tqdm(range(start_iter, 10000)):
    model.train()
    image = [x.to(device) for x in next(iterator_train)][0]
    batch, channels, height, width = image.shape
    mask = create_mask(batch_size, channels, height, width)
    gt = image
    
    image = image.to(device)
    mask = mask.to(device)
    gt = gt.to(device)
    
    
    output = model(image, mask)
    loss_dict = criterion(image, mask, output, gt)
    
    loss = 0.0
    for key, coef in LAMBDA_DICT.items():
        value = coef * loss_dict[key]
        loss += value
        if (i + 1) % log_interval == 0:
            writer.add_scalar('loss_{:s}'.format(key), value.item(), i + 1)
    
    import matplotlib.pyplot as plt

    if i%500 == 0:
        
        import matplotlib.pyplot as plt
        real = image[0, 0, :, :].cpu().detach().numpy() * 322
        fake = output[0, 0, :, :].detach().cpu().numpy()  * 322
        gt = gt[0, 0, :, :].cpu().detach().numpy() * 322
        concatenated_image = np.concatenate((real, fake), axis=2 if real.ndim == 3 else 1)
        concatenated_image = np.concatenate((concatenated_image, gt), axis=2 if real.ndim == 3 else 1)
        # Plotting
        plt.imshow(concatenated_image)
        plt.colorbar()
        plt.axis('off')  # Remove axes
        
        plt.savefig('output_image.png')
        plt.clf()
    
    if (i + 1) % save_model_interval == 0 or (i + 1) == max_iter:
        save_ckpt('{:s}/ckpt/{:d}.pth'.format(args.save_dir, i + 1),
                  [('model', model)], [('optimizer', optimizer)], i + 1)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print("loss = {:.2f}".format(loss.item()))
    
    
            
exit()




























create_data = Data_prep(data_path)

number_of_samples_t = 6000
number_of_samples_v = 700  
random_crop = False
shift_var = 24
norm  =  False
img_shape = (512, 512)
seed = 42

print('Creating Train data')
train_in,train_out = create_data(number_of_samples_t,\
            img_shape, shift_var,  random_crop)
print('Creating Val data')
val_in, val_out = create_data(number_of_samples_v, \
    img_shape, shift_var, seed=seed)

exit()
net = PConvUNet(23)
# Forward pass
image = torch.rand(1, 23, 256, 256)  # Batch size of 1
mask = torch.ones(1, 23, 256, 256)
mask[:, :, 28:36, 28:36] = 0  # Apply a simple square mask
outputs = net(image, mask)
print(outputs.shape)