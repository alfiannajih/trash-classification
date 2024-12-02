from transformers import PretrainedConfig
from typing import Dict

class MobileNetV3Config(PretrainedConfig):
    model_type = "mobilenetv3"

    def __init__(
        self,
        num_classes: int=6,
        id2label: Dict={
            0: "cardboard",
            1: "glass",
            2: "metal",
            3: "paper",
            4: "plastic",
            5: "trash"
        },
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.id2label = id2label
        self.num_classes = num_classes