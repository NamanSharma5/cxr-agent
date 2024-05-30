import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import TypeAlias, Union
import re
import sys


# Add the base_agent directory to sys.path
base_agent_dir = "/vol/biomedic3/bglocker/ugproj2324/nns20/cxr-agent/base_agent"
sys.path.insert(0, str(base_agent_dir))

from pathology_sets import Pathologies
from pathology_detector import CheXagentVisionTransformerPathologyDetector
from phrase_grounder import BioVilTPhraseGrounder
from generation_engine import GenerationEngine, Llama3Generation, CheXagentLanguageModelGeneration, GeminiFlashGeneration


pathology_dict_type: TypeAlias = dict[set]

DEVICE = None #"cuda:1"  
PATHOLOGY_DETECTION_THRESHOLD = 0.4
PHRASE_GROUNDING_THRESHOLD = 0.2

USER_PROMPT = "Just list the findings on the chest x-ray, nothing else. If there are no findings, just say that."
IGNORE_PATHOLOGIES = {} #{"Support Devices"}
DO_NOT_LOCALISE = {"Cardiomegaly"}

FILE_DUMP_RATE = 2

#VinDr paths
vinDr_test_ground_truth_path = Path("/vol/biomedic3/bglocker/ugproj2324/nns20/datasets/VinDr-CXR/test_set_three_splits/VinDr_test_test_split_with_one_hot_labels.csv")

vindr_pathologies = ["Aortic enlargement", "Atelectasis", "Calcification", "Cardiomegaly",
            "Clavicle fracture", "Consolidation", "Emphysema", "Enlarged PA",
            "ILD", "Infiltration", "Lung Opacity", "Lung cavity", "Lung cyst",
            "Mediastinal shift","Nodule/Mass", "Pleural effusion", "Pleural thickening",
            "Pneumothorax", "Pulmonary fibrosis","Rib fracture", "Other lesion",
            "No finding"] 

# CheXpert paths
cheXpert_test_ground_truth_path = Path("/vol/biomedic3/bglocker/ugproj2324/nns20/datasets/CheXpert/test.csv")
cheXpert_test_path = Path("/vol/biodata/data/chest_xray/CheXpert-v1.0-small/CheXpert-v1.0-small/test")
cheXpert_output_folder = Path("/vol/biomedic3/bglocker/ugproj2324/nns20/cxr-agent/base_agent/evaluation/cheXpert")

# CheXpert pathologies
cheXpert_pathologies = ['No Finding','Enlarged Cardiomediastinum','Cardiomegaly','Lung Opacity',
        'Lung Lesion','Edema','Consolidation','Pneumonia','Atelectasis','Pneumothorax',
        'Pleural Effusion','Pleural Other','Fracture','Support Devices']


def dump_outputs_to_files(model_outputs: dict, output_folder: Path):
    for image_id, model_dict in model_outputs.items():
        for model_name, model_output in model_dict.items():
            # replace any new lines in the model output with full stops
            model_output = model_output.replace("\n", ".")
            with open(output_folder / f"{model_name}.txt", "a") as output_file:
                output_file.write(f"{image_id},{model_output}\n")
    

if __name__ == "__main__":
    pathology_detector = CheXagentVisionTransformerPathologyDetector(pathologies=Pathologies.CHEXPERT, device=DEVICE)
    phrase_grounder = BioVilTPhraseGrounder(detection_threshold=PHRASE_GROUNDING_THRESHOLD, device = DEVICE)
    l3 = Llama3Generation(device = DEVICE)
    cheXagent_lm = CheXagentLanguageModelGeneration(pathology_detector.processor, pathology_detector.model, pathology_detector.generation_config, pathology_detector.device, pathology_detector.dtype)
    gemini = GeminiFlashGeneration()

    model_outputs = defaultdict(dict)
    with open(cheXpert_test_ground_truth_path) as chexpert_test_file:
        for index, line in enumerate(chexpert_test_file):
            if index <= 302:
                continue
            image_id = line.strip().split(",")[0]
            image_path = cheXpert_test_path / image_id
            pathology_confidences, localised_pathologies, chexagent_e2e = GenerationEngine.detect_and_localise_pathologies(
                image_path=image_path,
                pathology_detector=pathology_detector,
                phrase_grounder=phrase_grounder,
                pathology_detection_threshold = PATHOLOGY_DETECTION_THRESHOLD,
                ignore_pathologies=IGNORE_PATHOLOGIES,
                do_not_localise=DO_NOT_LOCALISE,
                prompt_for_chexagent_lm_output= USER_PROMPT,
            )
            
            model_outputs[image_id]['chexagent'] = chexagent_e2e

            gemini_system_prompt, gemini_image_context_prompt = gemini.generate_prompts(pathology_confidences, localised_pathologies)
            model_outputs[image_id]['gemini_agent'] = gemini.generate_model_output(gemini_system_prompt, gemini_image_context_prompt, user_prompt=USER_PROMPT)

            def run_chexagent_lm(pathology_confidences, localised_pathologies):
                system_prompt, image_context_prompt = GenerationEngine.generate_prompts(pathology_confidences, localised_pathologies)
                return cheXagent_lm.generate_model_output(system_prompt, image_context_prompt, user_prompt=USER_PROMPT)

            def run_llama3_agent(pathology_confidences, localised_pathologies):
                system_prompt, image_context_prompt = l3.generate_prompts(pathology_confidences, localised_pathologies)
                return l3.generate_model_output(system_prompt, image_context_prompt, user_prompt=USER_PROMPT)
            
            def run_gemini_agent(pathology_confidences, localised_pathologies):
                system_prompt, image_context_prompt = gemini.generate_prompts(pathology_confidences, localised_pathologies)
                return gemini.generate_model_output(system_prompt, image_context_prompt, user_prompt=USER_PROMPT)

            with ThreadPoolExecutor() as executor:       
                # Run cheXagent_lm and l3 in parallel
                future_chexagent_lm = executor.submit(run_chexagent_lm, pathology_confidences, localised_pathologies)
                future_llama3_agent = executor.submit(run_llama3_agent, pathology_confidences, localised_pathologies)

                for future in as_completed([future_chexagent_lm, future_llama3_agent]):
                    if future == future_chexagent_lm:
                        model_outputs[image_id]['chexagent_agent'] = future.result()
                    else:
                        model_outputs[image_id]['llama3_agent'] = future.result()

            if index % FILE_DUMP_RATE == 0:
                dump_outputs_to_files(model_outputs, cheXpert_output_folder)
                model_outputs = defaultdict(dict)
        
        dump_outputs_to_files(model_outputs, cheXpert_output_folder)