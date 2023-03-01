from torchvision.datasets import Cityscapes
from PIL import Image
from typing import Any, Callable, Dict, List, Optional, Union, Tuple
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2

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


transform=A.Compose(
                    [
                    A.Resize(256, 512),
                    A.HorizontalFlip(),
                    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                    ToTensorV2(),
                    ]
                    )