from dotenv import load_dotenv
import os
import wandb
import torch
from transformers.pipelines import PIPELINE_REGISTRY
from transformers import pipeline, AutoModelForImageClassification

from src.trash_classification.hf_model.config import MobileNetV3Config
from src.trash_classification.hf_model.model import MobileNetV3Model
from src.trash_classification.hf_model.pipeline import TrashClassificationPipeline

load_dotenv()

def get_weights(
    version: str
):
    api = wandb.Api()
    artifact = api.artifact(f"{os.getenv('ARTIFACT_PATH')}:{version}")

    local_path = artifact.download()
    state_dict = torch.load(os.path.join(local_path, "model.pth"))

    return state_dict

def deploy_to_hf(
    model_version: str="latest"
):
    config = MobileNetV3Config()
    model = MobileNetV3Model(config)

    state_dict = get_weights(version=model_version)
    for key in list(state_dict.keys()):
        state_dict[f"model.{key}"] = state_dict.pop(key)

    model.load_state_dict(state_dict)

    config.register_for_auto_class()
    model.register_for_auto_class("AutoModelForImageClassification")

    config.push_to_hub(os.getenv("HF_REPO"))
    model.push_to_hub(os.getenv("HF_REPO"))

    PIPELINE_REGISTRY.register_pipeline(
        "image-classification",
        pipeline_class=TrashClassificationPipeline,
        pt_model=AutoModelForImageClassification,
        type="image", 
    )

    pipe = pipeline(
        model=os.getenv("HF_REPO"),
        trust_remote_code=True
    )
    pipe.push_to_hub(os.getenv("HF_REPO"))

if __name__=="__main__":
    deploy_to_hf()