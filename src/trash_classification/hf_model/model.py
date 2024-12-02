from transformers import PreTrainedModel
from torchvision.models import mobilenet_v3_large, MobileNet_V3_Large_Weights
from torch import nn
import torch.nn.functional as F
from .config import MobileNetV3Config

class MobileNetV3Model(PreTrainedModel):
    config_class = MobileNetV3Config

    def __init__(self, config):
        super().__init__(config)
        self.model = mobilenet_v3_large(weights=MobileNet_V3_Large_Weights.DEFAULT)
        self.model.classifier = nn.Sequential(
            nn.Linear(960, 1280),
            nn.Hardswish(),
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(1280, config.num_classes),
        )
        
    def forward(self, tensor, labels=None):
        logits = self.model(tensor)
        
        if labels is not None:
            loss = F.cross_entropy(logits, labels)
            return {"loss": loss, "logits": logits}
        
        return {"logits": logits}