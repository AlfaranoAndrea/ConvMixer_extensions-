from torch.utils.data import DataLoader, Dataset
from pytorch_lightning import  LightningModule
import torchmetrics

from utils import encode_segmap , decode_segmap, get_nClasses
from models.convMixerUnet import ConvMixerUnet
from models.unet import Unet
import segmentation_models_pytorch as smp
import json
from torch import optim
from dataset import CityscapesDataset

class SemanticSegmentationModel(LightningModule):
    def __init__(self, architecture="ConvMixer"):
        self.parameter = json.load(open("config.json"))
        super(SemanticSegmentationModel,self).__init__()
        self.layer = ConvMixerUnet() #architecute
        self.criterion= smp.losses.DiceLoss(mode='multiclass')
        self.metrics = torchmetrics.JaccardIndex(num_classes=get_nClasses())
        self.train_class = CityscapesDataset('./data/', split='train', mode='fine',
                         target_type='semantic')
        self.val_class = CityscapesDataset('./data/', split='val', mode='fine',
                         target_type='semantic')
        self.test_class = CityscapesDataset('./data/', split='test', mode='fine',
                         target_type='semantic')

        if architecture== "ConvMixer":
            self.layer = ConvMixerUnet() #architecute
        elif architecture== "Unet":
            self.layer = Unet()
        else:
            raise 'undifined architecture provided'

    def process(self,image,segment):
        out=self(image)
        segment=encode_segmap(segment)
        loss=self.criterion(out,segment.long())
        iou=self.metrics(out,segment)
        return loss,iou

    def forward(self,x):
        return self.layer(x)

    def configure_optimizers(self):
        opt=optim.AdamW(self.parameters(), lr=self.parameter["lr"])
        return opt

    def train_dataloader(self):
        return DataLoader(self.train_class, batch_size=self.parameter["batch-size"], 
                          shuffle=True,num_workers=self.parameter["workers"],pin_memory=True)

    def training_step(self,batch,batch_idx):
        image,segment=batch
        loss,iou=self.process(image,segment)
        self.log('train_loss', loss,on_step=True, on_epoch=True,prog_bar=True)
        self.log('train_iou', iou,on_step=True, on_epoch=True,prog_bar=True)
        return loss

    def val_dataloader(self):
        return DataLoader(self.val_class, batch_size=self.parameter["batch-size"], 
                          shuffle=False,num_workers=self.parameter["workers"],pin_memory=True)

    def validation_step(self,batch,batch_idx):
        image,segment=batch
        loss,iou=self.process(image,segment)
        self.log('val_loss', loss,on_step=True, on_epoch=True,prog_bar=False)
        self.log('val_iou', iou,on_step=True, on_epoch=True,prog_bar=False)
        
    def test_dataloader(self):
        return DataLoader(self.test_class, batch_size=self.parameter["batch-size"]*8, 
                          shuffle=False,num_workers=self.parameter["workers"],pin_memory=True)

    def test_step(self,batch,batch_idx):
        image,segment=batch
        loss,iou=self.process(image,segment)
        self.log('test_loss', loss,on_step=False, on_epoch=True,prog_bar=False)
        self.log('test_iou', iou,on_step=False, on_epoch=True,prog_bar=False)
        return loss