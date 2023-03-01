#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data.dataloader import DataLoader

import torchvision
from torchvision import transforms
from torchvision import datasets

import numpy as np
import matplotlib.pyplot as plt
import cv2

from collections import Counter
import matplotlib.patches as patches
import torch.optim as optim

import tensorboard 

import pytorch_lightning as pl
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger

from  loss import YoloLoss
from  utilyties import funcion_ajuste_dataset

from PIL import Image, ImageDraw


# ### Parameters

# In[2]:


parameter= { "name": "ConvMixer",
             "batch-size": 2,
    
             "hdim":    768,
             "depth":   32, #32
             "psize":   7,
             "conv-ks": 7,
             "epochs" : 10,
             "lr-max": 2e-5,
             "workers": 32,
}
#Clases de VOC2007 to id numerico
class_to_id = {'aeroplane':0, 'bicycle':1, 'bird':2, 'boat':3, 'bottle':4,'bus':5, 'car':6, 'cat':7, 'chair':8, 'cow':9, 'diningtable':10,
                'dog':11, 'horse':12, 'motorbike':13, 'person':14, 'pottedplant':15,'sheep':16, 'sofa':17, 'train':18, 'tvmonitor':19}

#Id numerico to Clases de VOC2007
id_to_class = {i:c for c, i in class_to_id.items()}

S=7
C=20
B=2

torch.set_printoptions(threshold=10_000)


# ### DataModule

# In[3]:


class VOC(pl.LightningDataModule):
    def __init__ (self, batch_size=1):
        super().__init__()

        self.batch_size=parameter["batch-size"]   # Defining transforms to be applied on the data
        self.reshape_size= 448
        self.train_data = datasets.VOCDetection('./', year='2012', image_set='train',
                                                 transforms=self.transforms)      
        self.test_data = datasets.VOCDetection('./', year='2012', image_set='val',
                                                transforms=self.transforms)

        self.train_data_clean = datasets.VOCDetection('./', year='2012', image_set='train')
        self.test_data_clean = datasets.VOCDetection('./', year='2012', image_set='val')

    
    def transforms (self,img, ann ):
        transormation = transforms.Compose([
        transforms.Resize((self.reshape_size, self.reshape_size)), #Se fija el tamaño de las imagenes en 224x224 para evitar variaciones en las input a alas redes
        transforms.ToTensor(), #Se transforma a tensor
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
    
        box_batch = torch.Tensor(S,S,C+5*B).fill_(0)#Creo tensor de recuadros (7,7,30)

        w_img=int(img.size[0]) # Image width
        h_img=int(img.size[1]) # Image height

        img2 = transormation(img) #Proceso la imagen con la transformacion
        
        all_bbox = ann['annotation']['object'] #Obtengo todos los recuadros disponibles
        if type(all_bbox) == dict: # inconsistency in the annotation file
            all_bbox = [all_bbox]
        for bbox_idx, one_bbox in enumerate(all_bbox):
            bbox = one_bbox['bndbox'] #Obtengo parametros del recuadro x1 y1 x2 y2
            obj_cls = one_bbox['name'] #Obtengo la clase del objeto del rectangulo

            x = (int(bbox['xmax']) + int(bbox['xmin']))/(2*w_img) #Busco centro x del recuadro
            y = (int(bbox['ymax']) + int(bbox['ymin']))/(2*h_img) #Busco centro y del recuadro
            w = (int(bbox['xmax'])-int(bbox['xmin']))/w_img #Calculo ancho del recuadro
            h = (int(bbox['ymax'])-int(bbox['ymin']))/h_img #Calculo alto del recuadro
            class_label = class_to_id[obj_cls] #Paso a numero la clase del objeto

            i, j = int(S * y), int(S * x) #Obtengo pos de esq sup izquierda del cuadrante q contiene el centro
            x_cell, y_cell = S * x - j, S * y - i #Obtengo x e y relativo a esq sup izq del cuadrante correspondiente
            width_cell = w * S #Obtengo el ancho en relacion al ancho de la celda
            height_cell = h * S #Obtengo el alto en relación al alto de la celda

            if box_batch[i, j, 20] == 0: #Objeto no presente en ese cuadrante
                # Set that there exists an object
                box_batch[i, j, 20] = 1
                # Box coordinates
                box_coordinates = torch.tensor([x_cell, y_cell, width_cell, height_cell])
                #box_coordinates = torch.tensor([x, y, w, h])
                box_batch[i, j, 21:25] = box_coordinates
                # Set one hot encoding for class_label
                box_batch[i, j, class_label] = 1 #Pongo 1 en clase correspondiente

        return img2, box_batch
    
    def setup(self, stage=None):
        pass

    def get_trainset(self):
        return self.train_data
    
    def get_testset(self):
        return self.test_data
    
    def get_trainset_clean(self):
        return self.train_data_clean
    
    def get_testset_clean(self):
        return self.test_data_clean

    def train_dataloader(self):
        return DataLoader( self.train_data,batch_size=self.batch_size, pin_memory=True,num_workers=parameter["workers"]) 
  
    def test_dataloader(self):
        return DataLoader(self.test_data,batch_size=self.batch_size, pin_memory=True,num_workers=parameter["workers"]) 


# ### Loss function 

# In[4]:


def getCordinatesFromBBoxes (BBoxes):
    x1 = BBoxes[..., 0:1] - BBoxes[..., 2:3] / 2
    y1 = BBoxes[..., 1:2] - BBoxes[..., 3:4] / 2
    x2 = BBoxes[..., 0:1] + BBoxes[..., 2:3] / 2
    y2 = BBoxes[..., 1:2] + BBoxes[..., 3:4] / 2

    return x1, y1 , x2, y2


# In[5]:


def statistics (boxes_predictions, boxes_labels):
    box1_x1, box1_y1, box1_x2, box1_y2= getCordinatesFromBBoxes(boxes_predictions)
    box2_x1, box2_y1, box2_x2, box2_y2= getCordinatesFromBBoxes(boxes_labels)
    
    x1 = torch.max(box1_x1, box2_x1)
    y1 = torch.max(box1_y1, box2_y1)
    x2 = torch.min(box1_x2, box2_x2)
    y2 = torch.min(box1_y2, box2_y2)

    # .clamp(0) is for the case when they do not intersect
    intersection = (x2 - x1).clamp(0) * (y2 - y1).clamp(0)

    box1_area = abs((box1_x2 - box1_x1) * (box1_y2 - box1_y1))
    box2_area = abs((box2_x2 - box2_x1) * (box2_y2 - box2_y1))

    iou = intersection / (box1_area + box2_area - intersection + 1e-6)
    return iou
    


# ### Visualize boundingboxes and predictions

# In[6]:


def voc_wiewner(sample, predicted=None, S=7, B=2, C=20):
    image = sample[0] 
    annotation = sample [1]

    width,height= image.size
    print(width)
    print(height)
      
    all_bbox = annotation['annotation']['object'] #Obtengo todos los recuadros disponibles
    if type(all_bbox) == dict: # inconsistency in the annotation file
        all_bbox = [all_bbox]
    
    image1 = ImageDraw.Draw(image)  
    for bbox_idx, one_bbox in enumerate(all_bbox):
        bbox = one_bbox['bndbox'] #Obtengo parametros del recuadro x1 y1 x2 y2
        shape=[(int(bbox["xmin"]),int(bbox["ymin"])),(int(bbox['xmax']),int(bbox["ymax"]))]
        image1.rectangle(shape, outline ="red")

        obj_cls = one_bbox['name'] #Obtengo la clase del objeto del rectangulo
        image1.text(shape[0], obj_cls,align ="left", fill="red") 
    
    if(predicted is not None):
        print(predicted.size())
        if (predicted.size() != torch.Size([S, S, C + B * 5])):
            predicted = predicted.reshape(S, S, C + B * 5)
            print(predicted.size())

        for i in range(S):
            for j in range(S):

                if  predicted[i][j][20]>0.50: #Veo presencia de objeto
                    box = predicted[i][j][21:25].tolist()
                    obj_class = predicted[i][j][0:19].tolist()

                    upper_left_x = float(j*width/S) + float(box[0]*width/S) - float(box[2]*width/(2*S))
                    upper_left_y = float(i*height/S) + float(box[1]*height/S) - float(box[3]*height/(2*S))
                
                    bottom_left_x = float(j*width/S) + float(box[0]*width/S) + float(box[2]*width/(2*S))
                    bottom_left_y = float(i*height/S) + float(box[1]*height/S) + float(box[3]*height/(2*S))

                    shape= [(int(upper_left_x),int(upper_left_y)), (int(bottom_left_x),int(bottom_left_y))]
                    image1 .rectangle(shape, outline ="green")
                    
                    obj_cls = id_to_class[np.argmax(obj_class)]
                    image1 .text(shape[0], obj_cls,align ="left", fill="green") 

                if predicted[i][j][25]>0.50: #Veo presencia de objeto
                    box = predicted[i][j][26:30].tolist()
                    obj_class = predicted[i][j][0:19].tolist()

                    upper_left_x = float(j*width/S) + float(box[0]*width/S) - float(box[2]*width/(2*S))
                    upper_left_y = float(i*height/S) + float(box[1]*height/S) - float(box[3]*height/(2*S))
                
                    bottom_left_x = float(j*width/S) + float(box[0]*width/S) + float(box[2]*width/(2*S))
                    bottom_left_y = float(i*height/S) + float(box[1]*height/S) + float(box[3]*height/(2*S))

                    shape= [(int(upper_left_x),int(upper_left_y)), (int(bottom_left_x),int(bottom_left_y))]
                    image1 .rectangle(shape, outline ="green")
                    
                    obj_cls = id_to_class[np.argmax(obj_class)]
                    image1 .text(shape[0], obj_cls,align ="left", fill="green") 
  
    image.show()


# In[ ]:





# ### Net

# In[7]:


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x) + x


def ConvMixer(dim, depth, kernel_size, patch_size, n_classes):
    return nn.Sequential(
        nn.Conv2d(3, dim, kernel_size=patch_size, stride=patch_size),
        nn.ReLU(),
        nn.BatchNorm2d(dim),
        *[nn.Sequential(
                Residual(nn.Sequential(
                    nn.Conv2d(dim, dim, kernel_size, groups=dim, padding="same"),
                    nn.ReLU(),
                    nn.BatchNorm2d(dim)
                )),
                nn.Conv2d(dim, dim, kernel_size=1),
                nn.ReLU(),
                nn.BatchNorm2d(dim)
        ) for i in range(depth)],
         nn.AdaptiveAvgPool2d((1,1)),
         nn.Flatten(),
         nn.Linear(dim, n_classes)
    )


# In[10]:


#from symbol import parameters


class ConvMixerModule(pl.LightningModule):
    def __init__(self, checkpoint=None):
        super().__init__()
        
        self.model = ConvMixer(parameter['hdim'], parameter['depth'], patch_size=parameter['psize'], kernel_size=parameter['conv-ks'], n_classes=1000)
 
        self.criterion= YoloLoss()
        self.train_loss=0
        self.train_acc=0
        save = torch.load("convmixer_768_32_ks7_p7_relu.pth.tar")
        self.model.load_state_dict(save)
        self.model = nn.Sequential(*list(self.model.children())[:-3],
                nn.AdaptiveAvgPool2d((8,8)),
                nn.Flatten(),
                nn.Linear(parameter['hdim']*64, 496),
                nn.Dropout(0.0),
                nn.LeakyReLU(0.1),
                nn.Linear(496, S * S * (C + B * 5)))

    def forward (self, x):
        return self.model(x)
        
    def training_step(self, batch, batch_idx):
        img_batch, box_batch= batch    
        predictions = self(img_batch)
        loss =self.criterion(predictions,box_batch)

        #predictions = predictions.reshape(-1, S, S, C + B * 5)
        #print(statistics(predictions[..., 21:25], box_batch[..., 21:25]))

        self.log('train_loss', loss,on_step=True, on_epoch=True,prog_bar=True)
        return loss
    def validation_step(self,batch,batch_idx):
        img_batch, box_batch= batch    
        with torch.cuda.amp.autocast():
            predictions = self(img_batch)
            val_loss =self.criterion(predictions,box_batch)
        self.log('val_loss', val_loss,on_step=True, on_epoch=True,prog_bar=True)
        return val_loss
    def test_step(self, batch, batch_idx):
        img_batch, box_batch= batch    
        with torch.cuda.amp.autocast():
            predictions = self(img_batch)
            test_loss =self.criterion(predictions,box_batch)
            self.log('test_loss', test_loss,on_step=True, on_epoch=True,prog_bar=True)
        return test_loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=parameter['lr-max'], betas=(0.9, 0.999))
        return optimizer
   
    def save(self, path= '/models'):
        torch.save(self.state_dict(), path)


# ### train

# In[13]:


model  = ConvMixerModule.load_from_checkpoint("checkpoints/last.ckpt")
#model  = ConvMixerModule()
data = VOC()
ddp= DDPStrategy(find_unused_parameters=False)
checkpoint_callback = ModelCheckpoint(monitor='val_loss',dirpath='checkpoints',
                                         filename='file',save_last=True, every_n_epochs=1, save_top_k=5)

logger = TensorBoardLogger("tb_logs", name="my_model")
lr_monitor = LearningRateMonitor(logging_interval='epoch')
trainer = pl.Trainer(max_epochs=parameter["epochs"], auto_lr_find=False, auto_scale_batch_size=False,
                  gpus='0,1,2,3,4,5',precision=16,
                  callbacks=[checkpoint_callback, lr_monitor], 
                   strategy= ddp,
                  logger=logger
                 )
#trainer = pl.Trainer(gpus=1, max_epochs=parameter["epochs"], callbacks=[lr_monitor])
trainer.fit(model,data )


# In[ ]:


# trainer = pl.Trainer(max_epochs=parameter["epochs"], auto_lr_find=False, auto_scale_batch_size=False,
#                   gpus='0,2',precision=16,
#                   callbacks=[checkpoint_callback, lr_monitor], 
#                    strategy= ddp
#                  )


# ### testing

# In[ ]:





# In[ ]:


def testAndVisualize(n):
    dataset= VOC()
    original= dataset.get_trainset_clean()
    processed = dataset.get_trainset()
    tensor= processed[n][0]

    tensor= tensor.reshape(1, *tensor.size())
    tensor.to(torch.device("cuda"))

    label = model(tensor)
    label=label.reshape(1470)

    voc_wiewner(original[n],label)


# In[ ]:


#testAndVisualize(150)
#testAndVisualize(250)
#testAndVisualize(350)

#testAndVisualize(450)
testAndVisualize(570)
#testAndVisualize(650)


# In[ ]:




