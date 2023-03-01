from matplotlib import pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from utils import encode_segmap , decode_segmap, get_nClasses

import pytorch_lightning as pl
import albumentations as A
from albumentations.pytorch import ToTensorV2
import json
from dataset import CityscapesDataset
from segmentationModel import SemanticSegmentationModel



parameters = json.load(open("config.json"))
model  = SemanticSegmentationModel(architecture="Unet")
model.load_from_checkpoint("checkpoints/best.ckpt")

# trainer = pl.Trainer(
#             gpus=parameters["gpu"], 
#             )
test_class = CityscapesDataset('./data/', split='val', mode='fine',
                     target_type='semantic',transforms=model.transform)
test_loader=DataLoader(test_class, batch_size=12, 
                      shuffle=False)

transform=A.Compose(
[
    A.Resize(256, 512),
    A.HorizontalFlip(),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
]
)



#trainer.test(model)
model=model.cuda()
model.eval()
with torch.no_grad():
    for batch in test_loader:
        img,seg=batch
        output=model(img.cuda())
        break
print(img.shape,seg.shape,output.shape)   

from torchvision import transforms
inv_normalize = transforms.Normalize(
    mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
    std=[1/0.229, 1/0.224, 1/0.255]
)
sample=11
invimg=inv_normalize(img[sample])
outputx=output.detach().cpu()[sample]
encoded_mask=encode_segmap(seg[sample].clone()) #(256, 512)
decoded_mask=decode_segmap(encoded_mask.clone())  #(256, 512)
decoded_ouput=decode_segmap(torch.argmax(outputx,0))
fig,ax=plt.subplots(ncols=3,figsize=(16,50),facecolor='white')  
ax[0].imshow(np.moveaxis(invimg.numpy(),0,2)) #(3,256, 512)
#ax[1].imshow(encoded_mask,cmap='gray') #(256, 512)
ax[1].imshow(decoded_mask) #(256, 512, 3)
ax[2].imshow(decoded_ouput) #(256, 512, 3)
ax[0].axis('off')
ax[1].axis('off')
ax[2].axis('off')
ax[0].set_title('Input Image')
ax[1].set_title('Ground mask')
ax[2].set_title('Predicted mask')
plt.savefig('result.png',bbox_inches='tight')