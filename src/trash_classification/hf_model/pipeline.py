from transformers import Pipeline
from PIL import Image
import torchvision.transforms as v2
import torch
import torch.nn.functional as F

class TrashClassificationPipeline(Pipeline):
    def __init__(self, **kwargs):
        Pipeline.__init__(self, **kwargs)

        self.transform = v2.Compose([
            v2.CenterCrop(size=(224, 224)),
            v2.PILToTensor(), 
            v2.ConvertImageDtype(torch.float32),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def _sanitize_parameters(self, **kwargs):
        return {}, {}, {}

    def preprocess(self, inputs):
        img = Image.open(inputs)
        tensor = self.transform(img)
        tensor = tensor.unsqueeze(0)

        return tensor

    def _forward(self, tensor):
        self.model.eval()
        with torch.no_grad():
            out = self.model(tensor)["logits"]

        return out

    def postprocess(self, out):
        pred = F.softmax(out, dim=1).argmax(dim=1)[0]
        label = self.model.config.id2label[str(int(pred))]

        return label