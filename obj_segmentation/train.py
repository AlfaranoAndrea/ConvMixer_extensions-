import tensorboard 
import pytorch_lightning as pl
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks import EarlyStopping,ModelCheckpoint,LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
import json

from segmentationModel import SemanticSegmentationModel

parameters = json.load(open("config.json"))

model  = SemanticSegmentationModel(architecture="ConvMixer")
checkpoint_callback = ModelCheckpoint(
    monitor='val_loss',
    dirpath='checkpoints',
    filename='file',
    save_last=True, 
    every_n_epochs=1, 
    save_top_k=5)

logger = TensorBoardLogger(
            "tb_logs", 
            name="my_model")
lr_monitor = LearningRateMonitor(logging_interval='epoch')

if(len(parameters["gpu"]) <= 1):
    trainer = pl.Trainer(
            gpus=parameters["gpu"], 
            max_epochs=parameters["epochs"], 
            callbacks=[lr_monitor]
            )
else:
    ddp= DDPStrategy(find_unused_parameters=False)
    trainer = pl.Trainer(
            max_epochs=parameters["epochs"],
            auto_lr_find=False, 
            auto_scale_batch_size=False,
            gpus=parameters["gpu"],
            precision=16,
            callbacks=[checkpoint_callback, lr_monitor], 
            logger=logger,
            strategy= ddp
            )

trainer.fit(model )


# export PL_TORCH_DISTRIBUTED_BACKEND=gloo

