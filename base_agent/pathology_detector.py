from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

from PIL import Image
from pathlib import Path
import os
import torch
import torch.nn as nn
import numpy as np
os.environ['HF_HOME'] = '/vol/biomedic3/bglocker/ugproj2324/nns20/cxr-agent/.cheXagent_cache' ## THIS HAS TO BE BEFORE YOU IMPORT TRANSFORMERS
from transformers import AutoProcessor, AutoModelForCausalLM, GenerationConfig
from pathology_sets import Pathologies

PathologyConfidences = dict[str, float]

## PROBE ORDER super important  as was used to train the model
vindr_pathologies = ["Aortic enlargement", "Atelectasis", "Calcification", "Cardiomegaly",
              "Clavicle fracture", "Consolidation", "Emphysema", "Enlarged PA",
              "ILD", "Infiltration", "Lung Opacity", "Lung cavity", "Lung cyst",
              "Mediastinal shift","Nodule/Mass", "Pleural effusion", "Pleural thickening",
              "Pneumothorax", "Pulmonary fibrosis","Rib fracture", "Other lesion",
              "No finding"] 

cheXpert_pathologies = ['No Finding','Enlarged Cardiomediastinum','Cardiomegaly','Lung Opacity',
        'Lung Lesion','Edema','Consolidation','Pneumonia','Atelectasis','Pneumothorax',
        'Pleural Effusion','Pleural Other','Fracture','Support Devices']

vindr_layer_norm_weights = Path("/vol/biomedic3/bglocker/ugproj2324/nns20/CheXagent/model_inspection/post_layer_norm_best_vindr_model.pth")
cheXpert_layer_norm_weights = Path("/vol/biomedic3/bglocker/ugproj2324/nns20/CheXagent/model_inspection/post_layer_norm_best_chexpert_model.pth")

q_former_weights = Path("/vol/biomedic3/bglocker/ugproj2324/nns20/CheXagent/model_inspection/q_former_best_vindr_model.pth")

VISION_TRANSFORMERT_OUTPUT_EMBEDDING_SIZE = 1408
Q_FORMER_OUTPUT_EMBEDDING_SIZE = 98304


class PathologyDetector(ABC):

    @abstractmethod
    def detect_pathologies(self, image_path: Path, threshold:Optional[float]) -> PathologyConfidences:
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

    def __init__(self, pathologies: Pathologies = Pathologies.VINDR):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = torch.float16

        self.processor = AutoProcessor.from_pretrained("StanfordAIMI/CheXagent-8b", trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            "StanfordAIMI/CheXagent-8b", torch_dtype=self.dtype, trust_remote_code=True
        ).to(self.device)
        print("CheXagent Model loaded")
        self.generation_config = GenerationConfig.from_pretrained("StanfordAIMI/CheXagent-8b")
        
        if pathologies == Pathologies.VINDR:
            self.pathologies = vindr_pathologies
            self.linear_classifier = LinearClassifier(VISION_TRANSFORMERT_OUTPUT_EMBEDDING_SIZE, len(vindr_pathologies))
            self.linear_classifier.load_state_dict(torch.load(vindr_layer_norm_weights))
            self.linear_classifier.to(self.device)
        
        elif pathologies == Pathologies.CHEXPERT:
            self.pathologies = cheXpert_pathologies
            self.linear_classifier = LinearClassifier(98304, len(cheXpert_pathologies))
            self.linear_classifier.load_state_dict(torch.load(cheXpert_layer_norm_weights))
            self.linear_classifier.to(self.device)
        

    def detect_pathologies(self, image_path: Path,threshold: Optional[float] = None) -> PathologyConfidences:

        image = Image.open(image_path).convert("RGB")
        prompt = "NO PROMPT BEING USED"
        inputs = self.processor(
            images=image, text=f" USER: <s>{prompt} ASSISTANT: <s>", return_tensors="pt"
        ).to(device=self.device, dtype=self.dtype)

        embedding_output = self.model.generate(**inputs, return_vit_outputs = True,return_qformer_outputs = False, generation_config = self.generation_config)
        # map emebdding output to torch.float32
        embedding_output = embedding_output.to(torch.float)
        
        self.linear_classifier.eval()
        probe_logits = self.linear_classifier(torch.flatten(embedding_output))
        predictions = torch.sigmoid(probe_logits)

        pathology_confidences = dict(zip(self.pathologies, predictions.cpu().detach().numpy().flatten())) 

        if threshold is None:
            return pathology_confidences
        else:
            # filter out pathologies with confidence less than threshold
            return {pathology: confidence for pathology, confidence in pathology_confidences.items() if confidence > threshold}
    

# if __name__ == "__main__":
#     detector = CheXagentVisionTransformerPathologyDetector()

#     ### MIMIC-CXR ###

#     patient_id = '10012261'
#     study_id = "50349409"
#     mimic_cxr_dicom = Path('/vol/biodata/data/chest_xray/mimic-cxr-jpg/files')
#     # construct the path to the mimic-cxr folder ~/../../vol/biodata/data/chest_xray/mimic-cxr/{folder}/{patient_folder}
#     mimic_path = mimic_cxr_dicom / f"p{patient_id[:2]}" / f"p{patient_id}" / f"s{study_id}"

#     for image_path in mimic_path.iterdir():
#         print(detector.detect_pathologies(image_path))





    # cheXpert_small_path = Path("/vol/biodata/data/chest_xray/CheXpert-v1.0-small/CheXpert-v1.0-small/")
    # chexpert_test_csv_path = Path("/vol/biodata/data/chest_xray/CheXpert-v1.0-small/CheXpert-v1.0-small/test.csv")
    # chexpert_test_path = Path("/vol/biomedic3/bglocker/ugproj2324/nns20/datasets/CheXpert/small/")

    # with open(chexpert_test_csv_path, 'r') as f:
    #     lines = f.readlines()
    #     header = lines[0].split(",")[1:]
    #     # print(header)
    #     for i, line in enumerate(lines[1:]):
    #         if i % 1000 == 0:
    #             print(f"Collecting image {i}")

    #         image_path = line.split(",")[0]
    #         image_path = chexpert_test_path / image_path

    #         # print(line.split(",")[1:])
    #         print(dict(zip(header,line.split(",")[1:])))

            
    #         detector.detect_pathologies(image_path)
    #         break

        
