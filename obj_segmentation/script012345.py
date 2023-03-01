# %%
from matplotlib import pyplot as plt
from PIL import Image
import numpy as np
import torch
import torch.nn as nn
import tensorboard 

from pytorch_lightning import seed_everything, LightningModule, Trainer
import multiprocessing
import torchmetrics


from torch.utils.data import DataLoader, Dataset
from utils import encode_segmap , decode_segmap, get_nClasses

import pytorch_lightning as pl
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks import EarlyStopping,ModelCheckpoint,LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger

from einops import rearrange
import albumentations as A
from albumentations.pytorch import ToTensorV2

from typing import Any, Callable, Dict, List, Optional, Union, Tuple
from torchvision.datasets import Cityscapes

import segmentation_models_pytorch as smp

# %% [markdown]
# ## Parameters and utilyties

# %% [markdown]
# 

# %%
parameter= { "batch-size": 1,
             "lr": 2e-3,
             "workers": 8,
             "epochs" :10
}

# %% [markdown]
# ## Dataset

# %%
transform=A.Compose(
                    [
                    A.Resize(256, 512),
                    A.HorizontalFlip(),
                    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                    ToTensorV2(),
                    ]
                    )

# %%
class CityscapesDataset(Cityscapes):
    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        image = Image.open(self.images[index]).convert('RGB')

        targets: Any = []
        for i, t in enumerate(self.target_type):
            if t == 'polygon':
                target = self._load_json(self.targets[index][i])
            else:
                target = Image.open(self.targets[index][i])
            targets.append(target)
        target = tuple(targets) if len(targets) > 1 else targets[0]


        transformed=transform(image=np.array(image), mask=np.array(target))            
        return transformed['image'],transformed['mask']

# %% [markdown]
# ## Models

# %% [markdown]
# #### ConvMixer

# %%
class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x) + x

def ConvMixer(dim, depth, kernel_size=5, patch_size=2, in_layers=3):
    return nn.Sequential(
        nn.Conv2d(in_layers, dim, kernel_size=patch_size, stride=patch_size),
        nn.GELU(),
        nn.BatchNorm2d(dim),
        *[nn.Sequential(
                Residual(nn.Sequential(
                    nn.Conv2d(dim, dim, kernel_size, groups=dim, padding="same"),
                    nn.GELU(),
                    nn.BatchNorm2d(dim)
                )),
                nn.Conv2d(dim, dim, kernel_size=1),
                nn.GELU(),
                nn.BatchNorm2d(dim)
        ) for i in range(depth)]
    )

# %% [markdown]
# #### ConvMixer-Unet 

# %%
class ConvMixerUnet(nn.Module):
    def __init__(self ):
        super().__init__()
            #encoder side
        self.convmix1= ConvMixer(in_layers=3,dim=64, depth=8, kernel_size=9, patch_size=2)
        self.convmix2= ConvMixer(in_layers=64,dim=256, depth=8, kernel_size=9, patch_size=2)
        self.convmix3= ConvMixer(in_layers=256,dim=1024, depth=8, kernel_size=9, patch_size=2)
            # decoder side
        self.convmix4= ConvMixer(in_layers=512,dim=1024, depth=8, kernel_size=9, patch_size=2)
        self.convmix5= ConvMixer(in_layers=128,dim=320, depth=8, kernel_size=9, patch_size=2)
    def forward(self, x):
            #decoding blocks
        x1= self.convmix1(x)
        x2= self.convmix2(x1)
        x3= self.convmix3(x2)
        
            #x4 up resolution
        x3r= rearrange(x3, 'b (c1 c2 c3) h w-> b c1 (h c2) (w c3)', c2=2, c3=2)
            #concat with skip connection
        x3r= torch.concat([x3r, x2],1)
            # deconvolution 
        x4= self.convmix4(x3r)
            #second up resolution
        x4r= rearrange(x4, 'b (c1 c2 c3) h w-> b c1 (h c2) (w c3)', c2=4, c3=4)
            #concat with skip connection
        x4r= torch.concat([x4r, x1],1)
            #deconvolution
        x5= self.convmix5(x4r)
            #rearrange to output segmentated image
        x5r= rearrange(x5, 'b (c1 c2 c3) h w-> b c1 (h c2) (w c3)', c2=4, c3=4)
        return  x5r

# %% [markdown]
# #### Original Unet (from https://github.com/spctr01/UNet/blob/master/Unet.py)

# %%
#double 3x3 convolution 
def dual_conv(in_channel, out_channel):
    conv = nn.Sequential(
        nn.Conv2d(in_channel, out_channel, kernel_size=3),
        nn.ReLU(inplace= True),
        nn.Conv2d(out_channel, out_channel, kernel_size=3),
        nn.ReLU(inplace= True),
    )
    return conv


# crop the image(tensor) to equal size 
# as shown in architecture image , half left side image is concated with right side image
def crop_tensor(target_tensor, tensor):
    target_size = target_tensor.size()[2]
    tensor_size = tensor.size()[2]
    delta = tensor_size - target_size
    delta = delta // 2

    return tensor[:, :, delta:tensor_size- delta, delta:tensor_size-delta]

class Unet(nn.Module):
    def __init__(self):
        super(Unet, self).__init__()

        # Left side (contracting path)
        self.dwn_conv1 = dual_conv(1, 64)
        self.dwn_conv2 = dual_conv(64, 128)
        self.dwn_conv3 = dual_conv(128, 256)
        self.dwn_conv4 = dual_conv(256, 512)
        self.dwn_conv5 = dual_conv(512, 1024)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        #Right side  (expnsion path) 
        #transpose convolution is used showna as green arrow in architecture image
        self.trans1 = nn.ConvTranspose2d(1024,512, kernel_size=2, stride= 2)
        self.up_conv1 = dual_conv(1024,512)
        self.trans2 = nn.ConvTranspose2d(512,256, kernel_size=2, stride= 2)
        self.up_conv2 = dual_conv(512,256)
        self.trans3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride= 2)
        self.up_conv3 = dual_conv(256,128)
        self.trans4 = nn.ConvTranspose2d(128,64, kernel_size=2, stride= 2)
        self.up_conv4 = dual_conv(128,64)

        #output layer
        self.out = nn.Conv2d(64, 2, kernel_size=1)

    def forward(self, image):

        #forward pass for Left side
        x1 = self.dwn_conv1(image)
        x2 = self.maxpool(x1)
        x3 = self.dwn_conv2(x2)
        x4 = self.maxpool(x3)
        x5 = self.dwn_conv3(x4)
        x6 = self.maxpool(x5)
        x7 = self.dwn_conv4(x6)
        x8 = self.maxpool(x7)
        x9 = self.dwn_conv5(x8)
        

        #forward pass for Right side
        x = self.trans1(x9)
        y = crop_tensor(x, x7)
        x = self.up_conv1(torch.cat([x,y], 1))

        x = self.trans2(x)
        y = crop_tensor(x, x5)
        x = self.up_conv2(torch.cat([x,y], 1))

        x = self.trans3(x)
        y = crop_tensor(x, x3)
        x = self.up_conv3(torch.cat([x,y], 1))

        x = self.trans4(x)
        y = crop_tensor(x, x1)
        x = self.up_conv4(torch.cat([x,y], 1))
        
        x = self.out(x)
        
        return x

# %% [markdown]
# 
# ## model implementation in torch lightning

# %%

class SemanticSegmentation(LightningModule):
    def __init__(self):
        super(SemanticSegmentation,self).__init__()

        
        self.layer = ConvMixerUnet() #architecute
        self.criterion= smp.losses.DiceLoss(mode='multiclass')
        self.metrics = torchmetrics.JaccardIndex(num_classes=get_nClasses())
        self.train_class = CityscapesDataset('./data/', split='train', mode='fine',
                         target_type='semantic')
        self.val_class = CityscapesDataset('./data/', split='val', mode='fine',
                         target_type='semantic')
        self.test_class = CityscapesDataset('./data/', split='test', mode='fine',
                         target_type='semantic')
        self.lr=parameter["lr"]
        self.batch_size=parameter["batch-size"]
        self.numworker=parameter["workers"]

    def process(self,image,segment):
        out=self(image)
        segment=encode_segmap(segment)
        loss=self.criterion(out,segment.long())
        iou=self.metrics(out,segment)
        return loss,iou

    def forward(self,x):
        return self.layer(x)

    def configure_optimizers(self):
        opt=torch.optim.AdamW(self.parameters(), lr=self.lr)
        return opt

    def train_dataloader(self):
        return DataLoader(self.train_class, batch_size=self.batch_size, 
                          shuffle=True,num_workers=self.numworker,pin_memory=True)

    def training_step(self,batch,batch_idx):
        image,segment=batch
        loss,iou=self.process(image,segment)
        self.log('train_loss', loss,on_step=True, on_epoch=True,prog_bar=True)
        self.log('train_iou', iou,on_step=True, on_epoch=True,prog_bar=True)
        return loss

    def val_dataloader(self):
        return DataLoader(self.val_class, batch_size=self.batch_size, 
                          shuffle=False,num_workers=self.numworker,pin_memory=True)

    def validation_step(self,batch,batch_idx):
        image,segment=batch
        loss,iou=self.process(image,segment)
        self.log('val_loss', loss,on_step=True, on_epoch=True,prog_bar=False)
        self.log('val_iou', iou,on_step=True, on_epoch=True,prog_bar=False)
        
    def test_dataloader(self):
        return DataLoader(self.test_class, batch_size=self.batch_size*8, 
                          shuffle=False,num_workers=self.numworker,pin_memory=True)

    def test_step(self,batch,batch_idx):
        image,segment=batch
        loss,iou=self.process(image,segment)
        self.log('test_loss', loss,on_step=False, on_epoch=True,prog_bar=False)
        self.log('test_iou', iou,on_step=False, on_epoch=True,prog_bar=False)
        return loss

# %% [markdown]
# ## training 

# %%
model  = SemanticSegmentation()
ddp= DDPStrategy(find_unused_parameters=False)
checkpoint_callback = ModelCheckpoint(monitor='val_loss',dirpath='checkpoints',
                                          filename='file',save_last=True, every_n_epochs=1, save_top_k=5)
logger = TensorBoardLogger("tb_logs", name="my_model")
lr_monitor = LearningRateMonitor(logging_interval='epoch')
trainer = pl.Trainer(max_epochs=30, auto_lr_find=False, auto_scale_batch_size=False,
                   gpus='0,1,2,3,4,5,6,7',precision=16,
                   callbacks=[checkpoint_callback, lr_monitor], 
                    strategy= ddp,
                   logger=logger
                  )
#trainer = pl.Trainer(gpus=1, max_epochs=parameter["epochs"], callbacks=[lr_monitor])
trainer.fit(model )

# %% [markdown]
# ## Testing

# %%


