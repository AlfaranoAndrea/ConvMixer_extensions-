import time
import torch
import itertools
import torchvision
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.nn.functional as F
import cv2
import os
from os import path
from torchvision import transforms
from torchvision import datasets
from torchvision.datasets import ImageFolder
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import RandomSampler
from sklearn.metrics import accuracy_score,f1_score,precision_score,recall_score
import matplotlib.patches as patches
from collections import Counter
import pytorch_lightning as pl
import torch.optim as optim

class_to_id = {'aeroplane':0, 'bicycle':1, 'bird':2, 'boat':3, 'bottle':4,'bus':5, 'car':6, 'cat':7, 'chair':8, 'cow':9, 'diningtable':10,
                'dog':11, 'horse':12, 'motorbike':13, 'person':14, 'pottedplant':15,'sheep':16, 'sofa':17, 'train':18, 'tvmonitor':19}
#Id numerico to Clases de VOC2007
id_to_class = {i:c for c, i in class_to_id.items()}
#Función para ajuste del dataset de imágenes de VOC2007

def funcion_ajuste_dataset(batch, reshape_size=448,S=7, B=2, C=20): #Función para ajuste del dataset 448
    transormation = transforms.Compose([
      transforms.Resize((reshape_size, reshape_size)), #Se fija el tamaño de las imagenes en 224x224 para evitar variaciones en las input a alas redes
      transforms.ToTensor(), #Se transforma a tensor
      transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
      ])
    
    batch_size = len(batch)
    
    #Se prepara batch de imagenes
    img_batch = torch.zeros(batch_size, 3, reshape_size, reshape_size) #Creo tensor de imagenes (B,3,448,448)

  
    #Se prepara batch de recuadros
    box_batch = torch.Tensor(batch_size,S,S,C+5*B).fill_(0)#Creo tensor de recuadros (B,7,7,30)
    

    for b_i in range(batch_size): #Se recorre el batch
      img, ann = batch[b_i] #Se obtiene Imagen,Annotation
      w_img=int(img.size[0]) # Image width
      h_img=int(img.size[1]) # Image height

      img_batch[b_i] = transormation(img) #Proceso la imagen con la transformacion
      
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

        if box_batch[b_i][i, j, 20] == 0: #Objeto no presente en ese cuadrante
          # Set that there exists an object
          box_batch[b_i][i, j, 20] = 1
          # Box coordinates
          box_coordinates = torch.tensor([x_cell, y_cell, width_cell, height_cell])
          #box_coordinates = torch.tensor([x, y, w, h])
          box_batch[b_i][i, j, 21:25] = box_coordinates
          # Set one hot encoding for class_label
          box_batch[b_i][i, j, class_label] = 1 #Pongo 1 en clase correspondiente

    return img_batch, box_batch



