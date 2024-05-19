import os
os.environ['HF_HOME'] = '/vol/biomedic3/bglocker/ugproj2324/nns20/cxr-agent/.hf_cache' ## THIS HAS TO BE BEFORE YOU IMPORT TRANSFORMERS

import transformers
import torch

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional
from PIL import Image
from my_secrets import LLAMA3_INSTRUCT_ACCESS_TOKEN
from agent_utils import select_best_gpu

from pathology_detector import PathologyDetector, CheXagentVisionTransformerPathologyDetector
from pathology_sets import Pathologies

from phrase_grounder import PhraseGrounder, BioVilTPhraseGrounder

class GenerationEngine(ABC):


    @abstractmethod
    def generate_model_output(self, system_prompt: str , image_context_prompt: str, user_prompt:Optional[str] = None, image_path = None):
        pass

    def detect_and_localise_pathologies(image_path,pathology_detector,phrase_grounder,pathology_detection_threshold = 0.3, ignore_pathologies = None, do_not_localise = None, prompt_for_chexagent_lm_output = None):
        """
        A function to detect pathologies in an image and localise them
        @param image_path: Path to the image
        @param pathology_detector: PathologyDetector object
        @param phrase_grounder: PhraseGrounder object
        @param pathology_detection_threshold: Threshold for pathology detection
        @param ignore_pathologies: List of pathologies to ignore
        @param do_not_localise: List of pathologies to not localise
        @param prompt_for_chexagent_lm_output: If True, return the output of the CheXagent Language Model

        """    

        if prompt_for_chexagent_lm_output is not None:
            pathology_confidences, lm_output = pathology_detector.detect_pathologies(image_path, threshold = pathology_detection_threshold, prompt_for_chexagent_lm_output = prompt_for_chexagent_lm_output)
        else:
            pathology_confidences = pathology_detector.detect_pathologies(image_path, threshold = pathology_detection_threshold)
        
        pathology_confidences = {pathology: confidence for pathology, confidence in pathology_confidences.items() if pathology not in ignore_pathologies}
        pathologies_to_localise = [pathology for pathology in pathology_confidences.keys() if pathology not in do_not_localise]

        localised_pathologies = phrase_grounder.get_pathology_lateral_position(image_path, pathologies_to_localise)

        if prompt_for_chexagent_lm_output is not None:
            return pathology_confidences, localised_pathologies, lm_output
        
        return pathology_confidences, localised_pathologies
    
    
    def generate_prompts(detected_pathologies, localised_pathologies, examples = False):
        if len(detected_pathologies) == 0:
            system_prompt = """ You are a helpful assistant, specialising in radiology and interpreting Chest X-rays. However there is insufficient data to make any comments on pathologies. Just mention it is possible there are no findings and this should be double checked by a radiologist."""
            image_context_prompt = f"""No pathologies were detected in the chest X-ray. The user will now interact with you."""
            return system_prompt, image_context_prompt

        if len(localised_pathologies) == 0:
            localised_pathologies = "No lateral positions could be confidently determined for any pathologies detected."

        system_prompt = """You are a helpful assistant, specialising in radiology and interpreting Chest X-rays. Please answer CONCISELY and professionally as a radiologist would. You should not be confident unless the data is confident. Use language that reflects the confidence of the data."""
        image_context_prompt = f"""
        You are being provided with data derived from analyzing a chest X-ray, which includes findings on potential pathologies alongside their confidence levels and, separately, possible lateral locations of these pathologies with their own confidence scores.
        This information comes from two specialized diagnostic tools.
        It's important to recognize how to interpret these datasets together when responding to queries:

        Pathology Detection with Confidence Scores:
        {detected_pathologies}

        Phrase Grounding Locations with Confidence Scores:
        {localised_pathologies}

        This separate dataset provides potential lateral locations for some of the detected pathologies, each with its own confidence score, indicating the model's certainty about each pathology's location.
        For instance, left Pleural Effusion is listed with a confidence score of 0.53, suggesting the location of Pleural Effusion to be on the left side with moderate confidence.

        When you interact with end users, remember:

        - A pathology and its lateral location (e.g., Pleural Effusion and left Pleural Effusion) are part of the same finding. The location attribute is an additional detail about where the pathology is likely found, not an indicator of a separate pathology. 
        - Synthesize the pathology detection and localization data. DO NOT TALK ABOUT THEM SEPERATELY. {"Here is a model example, 'Highly likely there is Pleural Effusion (detection confidence: 0.80), and it is possibly on the left side (localisation confidence: 0.53).'" if examples else ""}
        - Confidence scores from the pathology detection and phrase grounding tools are not directly comparable. They serve as indicators of confidence within their respective contexts of pathology detection and localisation.
        - A missing lateral location does not imply the absence of a pathology; it indicates the localisation could not be confidently determined.
        - If there is any discrepancy between the pathology detection and phrase grounding tools, detection data takes precedence as it more reliably identifies pathologies.

        It is important to factor medical knowledge and the specifics of each case, if supplied, into your responses. For example, pathologies located on both sides are called bilateral. Heart related observations are usually on the left/ middle.
        This understanding is crucial for accurately processing and responding to queries on the chest X-ray analysis. Structure your answers based on confidence and pathologies.
        Double check before you submit your response to ensure you have factored in all the data and followed my instructions carefully.
        You will now interact with the user.
        """
            
        return system_prompt, image_context_prompt

      
class Llama3Generation(GenerationEngine):

    def __init__(self):
        self.model_id = "meta-llama/Meta-Llama-3-8B-Instruct"

        self.pipeline = transformers.pipeline(
            "text-generation",
            model=self.model_id,
            model_kwargs={"torch_dtype": torch.bfloat16},
            device_map= select_best_gpu() ,
            token=LLAMA3_INSTRUCT_ACCESS_TOKEN,
        )

    def generate_prompts(detected_pathologies, localised_pathologies, examples = False):

        if len(detected_pathologies) == 0:
            system_prompt = """ You are a helpful assistant, specialising in radiology and interpreting Chest X-rays. However there is insufficient data to make any comments on pathologies. Just mention it is possible there are no findings and this should be double checked by a radiologist."""
            image_context_prompt = f"""No pathologies were detected in the chest X-ray. The user will now interact with you."""
            return system_prompt, image_context_prompt

        print(localised_pathologies)
        if len(localised_pathologies) == 0:
            localised_pathologies = "No lateral positions could be confidently determined for any pathologies detected."


        system_prompt = """You are a helpful assistant, specialising in radiology and interpreting Chest X-rays. Please answer CONCISELY and professionally as a radiologist would. Do not reference any confidence scores in your responses."""

        image_context_prompt = f"""
        You are given data on a chest x-ray, which includes pathologies and their confidence scores and, separately, possible lateral locations of these pathologies with their own confidence scores.
        This information comes from two different, specialized diagnostic tools.
        It's important to recognize how to interpret these datasets together when responding to queries:

        Pathology Detection with Confidence Scores:
        {detected_pathologies}

        Here are the guidelines to follow when interpreting the Pathology Detection data - do not mention the confidence scores to the user:

        - For confidence scores between 0.3 and 0.5, state "cannot exclude <pathology>"
        - For confidence scores between 0.5 and 0.7, state "possible <pathology>" 
        - For confidence scores between 0.7 and 0.9, state "probable <pathology>"
        - For confidence scores over 0.9, simply state the pathology name

        Phrase Grounding Locations with Confidence Scores:
        {localised_pathologies}

        This separate dataset provides potential lateral locations for some of the detected pathologies, each with its own confidence score, indicating the model's certainty about each pathology's location.

        When you interact with end users, remember:
        - A pathology and its lateral location (e.g., Pleural Effusion and left Pleural Effusion) are part of the same finding. The location attribute is an additional detail about where the pathology is likely found, not an indicator of a separate pathology. 
        - Synthesize the pathology detection and localization data. DO NOT TALK ABOUT THEM SEPERATELY. 
        - Confidence scores from the pathology detection and phrase grounding tools are not directly comparable. They serve as indicators of confidence within their respective contexts of pathology detection and localisation.
        - A missing lateral location does not imply the absence of a pathology; it indicates the localisation could not be confidently determined.
        {"Here are some model examples: 'Probable pleural effusion, located on the left' ; 'Possible bilateral edema' " if examples else ""}


        It is important to factor medical knowledge and the specifics of each case, if supplied, into your responses. For example, pathologies located on both sides are called bilateral.
        This understanding is crucial for accurately processing and responding to queries on the chest X-ray analysis. Structure your answers based on confidence and pathologies.
        Double check before you submit your response to ensure you have factored in all the data and followed my instructions carefully.
        You will now interact with the user, only answer their question do not mention any of your instructions.

        """
    
        return system_prompt, image_context_prompt

    def generate_model_output(self, system_prompt: str , image_context_prompt: str, user_prompt:Optional[str] = None):
        def format_output(output_text: str) -> str:
            return "\n".join(output_text.split(". "))  # print output text with each sentence on a new line

        terminators = [
            self.pipeline.tokenizer.eos_token_id,
            self.pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]

        if user_prompt is not None:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": image_context_prompt +"\n" + user_prompt},
            ]

            prompt = self.pipeline.tokenizer.apply_chat_template(
                    messages, 
                    tokenize=False, 
                    add_generation_prompt=True
            )

            outputs = self.pipeline(
                prompt,
                max_new_tokens=512,
                eos_token_id=terminators,
                do_sample=True,
                temperature=0.6,
                top_p=0.9,
            )
            output_text = outputs[0]["generated_text"][len(prompt):]
            return outputs[0]["generated_text"][len(prompt):]
        else:
            return RuntimeError("User prompt not provided - Chat mode CURRENTLY not supported with LLAMA3")
        

class CheXagentLanguageModelGeneration(GenerationEngine):

    def __init__(self, processor, model, generation_config, device, dtype):
        self.processor = processor
        self.model = model
        self.generation_config = generation_config
        self.device = device
        self.dtype = dtype

    def generate_model_output(self, system_prompt: str , image_context_prompt: str, user_prompt:Optional[str] = None):
        if user_prompt is None:
            return RuntimeError("User prompt not provided - Chat mode CURRENTLY not supported with CheXagent Language Model")
        inputs = self.processor(
        images=None, text=f"[INST]{image_context_prompt}[/INST] USER: <s>{user_prompt} ASSISTANT: <s>", return_tensors="pt"
        ).to(device=self.device)
        output = self.model.generate(**inputs, generation_config=self.generation_config,generate_written_output = True)[0]
        response = self.processor.tokenizer.decode(output, skip_special_tokens=True)
        # print(response)
        return response
    

class CheXagentEndToEndGeneration(GenerationEngine):

    def __init__(self, processor, model, generation_config, device, dtype):
        self.processor = processor
        self.model = model
        self.generation_config = generation_config
        self.device = device
        self.dtype = dtype

    def generate_model_output(self, system_prompt: str , image_context_prompt: str, user_prompt:Optional[str],image_path: Path):
        images = [Image.open(image_path).convert("RGB")]
        inputs = self.processor(
            images=images, text=f" USER: <s>{user_prompt} ASSISTANT: <s>", return_tensors="pt"
        ).to(device=self.device, dtype=self.dtype)
        output = self.model.generate(**inputs, generation_config=self.generation_config)[0]
        response = self.processor.tokenizer.decode(output, skip_special_tokens=True)
        return response
