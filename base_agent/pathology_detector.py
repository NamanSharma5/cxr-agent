from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional
from agent_utils import select_best_gpu

from PIL import Image
from pathlib import Path
import os
import torch
import torch.nn as nn
import numpy as np
os.environ['HF_HOME'] = '/vol/biomedic3/bglocker/ugproj2324/nns20/cxr-agent/.hf_cache' ## THIS HAS TO BE BEFORE YOU IMPORT TRANSFORMERS
from transformers import AutoProcessor, AutoModelForCausalLM, GenerationConfig
from pathology_sets import Pathologies

PathologyConfidences = dict[str, float]

## PROBE ORDER super important  as was used to train the model
vindr_pathologies = ["Aortic enlargement", "Atelectasis", "Calcification", "Cardiomegaly",
              "Clavicle fracture", "Consolidation", "Emphysema", "Enlarged PA",
              "ILD", "Infiltration", "Lung Opacity", "Lung cavity", "Lung cyst",
              "Mediastinal shift","Nodule/Mass", "Pleural effusion", "Pleural thickening",
              "Pneumothorax", "Pulmonary fibrosis","Rib fracture", "Other lesion",
              "No Finding"] 

cheXpert_pathologies = ['No Finding','Enlarged Cardiomediastinum','Cardiomegaly','Lung Opacity',
        'Lung Lesion','Edema','Consolidation','Pneumonia','Atelectasis','Pneumothorax',
        'Pleural Effusion','Pleural Other','Fracture','Support Devices']

vindr_layer_norm_weights = Path("/vol/biomedic3/bglocker/ugproj2324/nns20/CheXagent/model_inspection/post_layer_norm_best_vindr_model.pth")
cheXpert_layer_norm_weights = Path("/vol/biomedic3/bglocker/ugproj2324/nns20/CheXagent/model_inspection/post_layer_norm_best_chexpert_model.pth")

q_former_weights = Path("/vol/biomedic3/bglocker/ugproj2324/nns20/CheXagent/model_inspection/q_former_best_vindr_model.pth")

VISION_TRANSFORMERT_OUTPUT_EMBEDDING_SIZE = 1408
Q_FORMER_OUTPUT_EMBEDDING_SIZE = 98304
DECIMALS = 2


class PathologyDetector(ABC):

    @abstractmethod
    def detect_pathologies(self, image_path: Path, threshold:Optional[float], prompt_for_chexagent_lm_output = None) -> PathologyConfidences:
        pass

class LinearClassifier(nn.Module):
    def __init__(self, num_features, num_classes):
        super(LinearClassifier, self).__init__()
        self.linear = nn.Linear(num_features, num_classes)
    
    def forward(self, x):
        x = self.linear(x)
        # Shape of x becomes [batch_size, num_classes]
        return x

class CheXagentVisionTransformerPathologyDetector(PathologyDetector):

    def __init__(self, pathologies: Pathologies = Pathologies.VINDR, device = None):
        if device is None:
            self.device = select_best_gpu()
        else:
            self.device = device
        self.dtype = torch.float16

        local_model_path = Path("/vol/biomedic3/bglocker/ugproj2324/nns20/cxr-agent/.hf_cache/hub/models--StanfordAIMI--CheXagent-8b/snapshots/4934e91451945c8218c267aae9c34929a7677829")
        self.processor = AutoProcessor.from_pretrained(local_model_path, trust_remote_code=True, revision="4934e91451945c8218c267aae9c34929a7677829")#, revision="4934e91451945c8218c267aae9c34929a7677829")
        self.model = AutoModelForCausalLM.from_pretrained(
            local_model_path, torch_dtype=self.dtype, trust_remote_code=True, revision="4934e91451945c8218c267aae9c34929a7677829"
        ).to(self.device)
        print("CheXagent Model loaded")
        self.generation_config = GenerationConfig.from_pretrained(local_model_path,revision="4934e91451945c8218c267aae9c34929a7677829")
        
        if pathologies == Pathologies.VINDR:
            self.pathologies = vindr_pathologies
            self.linear_classifier = LinearClassifier(VISION_TRANSFORMERT_OUTPUT_EMBEDDING_SIZE, len(vindr_pathologies))
            self.linear_classifier.load_state_dict(torch.load(vindr_layer_norm_weights))
            self.linear_classifier.to(self.device)
        
        elif pathologies == Pathologies.CHEXPERT:
            self.pathologies = cheXpert_pathologies
            self.linear_classifier = LinearClassifier(VISION_TRANSFORMERT_OUTPUT_EMBEDDING_SIZE, len(cheXpert_pathologies))
            self.linear_classifier.load_state_dict(torch.load(cheXpert_layer_norm_weights))
            self.linear_classifier.to(self.device)
        

    def detect_pathologies(self, image_path: Path,threshold: Optional[float] = None, decimals = DECIMALS, prompt_for_chexagent_lm_output = None) -> PathologyConfidences:

        image = Image.open(image_path).convert("RGB")
        if prompt_for_chexagent_lm_output is None:
            prompt = "NO PROMPT BEING USED"
        
            inputs = self.processor(
                images=image, text=f" USER: <s>{prompt} ASSISTANT: <s>", return_tensors="pt"
            ).to(device=self.device, dtype=self.dtype)

            embedding_output = self.model.generate(**inputs, return_vit_outputs = True,return_qformer_outputs = False, generation_config = self.generation_config)
        else:
            prompt = prompt_for_chexagent_lm_output
            inputs = self.processor(
                images=image, text=f" USER: <s>{prompt} ASSISTANT: <s>", return_tensors="pt"
            ).to(device=self.device, dtype=self.dtype)
            embedding_output, lm_output = self.model.generate(**inputs, return_vit_outputs = True,return_qformer_outputs = False, generate_written_output = True, generation_config = self.generation_config)
            lm_output = self.processor.tokenizer.decode(lm_output[0], skip_special_tokens=True)
        #  map emebdding output to torch.float32
        embedding_output = embedding_output.to(torch.float)
        
        self.linear_classifier.eval()
        probe_logits = self.linear_classifier(torch.flatten(embedding_output))
        predictions = torch.sigmoid(probe_logits)
        
        # round confidences to 2 decimal places
        predictions = torch.round(predictions, decimals=decimals)

        pathology_confidences = dict(zip(self.pathologies, predictions.cpu().detach().numpy().flatten())) 

        if threshold is not None:
           pathology_confidences = {pathology: confidence for pathology, confidence in pathology_confidences.items() if confidence > threshold}

        if prompt_for_chexagent_lm_output is not None:
            return pathology_confidences, lm_output
        
        return pathology_confidences


if __name__ == "__main__":
    detector = CheXagentVisionTransformerPathologyDetector()

    chexpert_test_csv_path = Path("/vol/biodata/data/chest_xray/CheXpert-v1.0-small/CheXpert-v1.0-small/test.csv")
    chexpert_test_path = Path("/vol/biomedic3/bglocker/ugproj2324/nns20/datasets/CheXpert/small/")

    with open(chexpert_test_csv_path, 'r') as f:
        lines = f.readlines()
        header = lines[0].split(",")[1:]
        # print(header)
        for i, line in enumerate(lines[1:]):
            if i % 1000 == 0:
                print(f"Collecting image {i}")

            image_path = line.split(",")[0]
            image_path = chexpert_test_path / image_path
            
            print(detector.detect_pathologies(image_path, prompt_for_chexagent_lm_output="What are the findings?"))
            break

        
