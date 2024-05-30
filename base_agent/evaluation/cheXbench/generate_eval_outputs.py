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
PHRASE_GROUNDING_THRESHOLD = 0.1

USER_PROMPT = "Just list the findings on the chest x-ray, nothing else. If there are no findings, just say that."
IGNORE_PATHOLOGIES = {} #{"Support Devices"}
DO_NOT_LOCALISE = {"Cardiomegaly"}

FILE_DUMP_RATE = 2


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
    
    itr_chexbench = Path("/vol/biomedic3/bglocker/ugproj2324/nns20/cxr-agent/evaluation_datasets/CheXbench/image_text_reasoning_task")
    openi_dataset_path = Path('/vol/biodata/data/chest_xray/OpenI/NLMCXR_png')
    chexbench_output_folder = Path("/vol/biomedic3/bglocker/ugproj2324/nns20/cxr-agent/base_agent/evaluation/cheXbench")

    specific_pathologies = {"opacity","atelectasis","effusion"}

    with open(itr_chexbench) as f:
        f.readline() # skip header
        for index, line in enumerate(f):
            image_id = line.split(',')[2]
            image_path = openi_dataset_path / image_id
            question = line.split(',')[3]
            option_1 = line.split(',')[5]
            option_2 = line.split(',')[6]

            if not (("left" in option_1 and "right" in option_2) or ("right" in option_1 and "left" in option_2)):
                continue   

            # for each word in option 1 , check if it is in the specific pathologies list, if any word is 
            # in the list, then we will use that option as the user prompt
            found = False
            for word in option_1.split(" "):
                if word in specific_pathologies:
                    found = True

            if not found:
                continue

            USER_PROMPT = f"{question} Option 1:{option_1} or Option 2: {option_2}."
            # image_text_reasoning_responses.append(f"{image_id},{response}")

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
                dump_outputs_to_files(model_outputs, chexbench_output_folder)
                model_outputs = defaultdict(dict)
        
        dump_outputs_to_files(model_outputs, chexbench_output_folder)