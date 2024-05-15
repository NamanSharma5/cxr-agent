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

    
    def contextualise_model(image_path: Path, pathology_detector: PathologyDetector, phrase_grounder: Optional[PhraseGrounder] = None, examples = True,  prompt_for_chexagent_lm_output = None) -> str:
        
        pathology_detection_threshold = 0.5

        if pathology_detector is not None:
            if prompt_for_chexagent_lm_output is not None:
                pathology_confidences, lm_output = pathology_detector.detect_pathologies(image_path, threshold = pathology_detection_threshold, prompt_for_chexagent_lm_output = prompt_for_chexagent_lm_output)
            else:
                pathology_confidences = pathology_detector.detect_pathologies(image_path, threshold = pathology_detection_threshold)
            # print(pathology_confidences)
        else:
            return RuntimeError("Pathology detector not provided")
        
        #### PROMPT PIPELINE ###
        if len(pathology_confidences) == 0:
            system_prompt = """ You are a helpful assistant, specialising in radiology and interpreting Chest X-rays. However there is insufficient data make any comments. Just mention it is possible there are no findings and this should be double checked by a radiologist."""
            image_context_prompt = f"""No pathologies were detected in the chest X-ray image. The user will now interact with you."""
            return system_prompt, image_context_prompt
        

        system_prompt = """You are a helpful assistant, specialising in radiology and interpreting Chest X-rays. You MUST answer CONCISELY and professionally as a radiologist would. You should not be confident unless the data is confident. Use language that reflects the confidence of the data."""
        image_context_prompt_final_part = """
            This understanding is crucial for accurately processing and responding to queries on the chest X-ray analysis. Structure your answers based on confidence and pathologies.
            Double check before you submit your response to ensure you have factored in all the data and followed my instructions carefully.
            You will now interact with the user.
            """

        if phrase_grounder is None:
            image_context_prompt = f"""
                You are being provided with data derived from analyzing a chest X-ray, which includes findings on potential pathologies alongside their confidence levels.
                This information comes from a specialized diagnostic tools.
                It's important to recognize how to interpret this when responding to queries:

                Pathology Detection with Confidence Scores:
                {pathology_confidences}
                Please note the closer the value to 1, the more likely the pathology is present in the image. 

                It is important to factor medical knowledge and the specifics of each case, if supplied, into your responses.
                {image_context_prompt_final_part}
                """

        else:
            pathologies = [pathology for pathology, confidence in pathology_confidences.items() if confidence > pathology_detection_threshold]
            grounded_pathologies_confidences = phrase_grounder.get_pathology_lateral_position(image_path, pathologies)
            # print(grounded_pathologies_confidences)
            if len(grounded_pathologies_confidences) == 0:
                grounded_pathologies_confidences = "No lateral positions could be confidently determined for any pathologies detected."

            image_context_prompt = f"""
            You are being provided with data derived from analyzing a chest X-ray, which includes findings on potential pathologies alongside their confidence levels and, separately, possible lateral locations of these pathologies with their own confidence scores.
            This information comes from two specialized diagnostic tools.
            It's important to recognize how to interpret these datasets together when responding to queries:

            Pathology Detection with Confidence Scores:
            {pathology_confidences}

            Phrase Grounding Locations with Confidence Scores:
            {grounded_pathologies_confidences}

            This separate dataset provides potential lateral locations for some of the detected pathologies, each with its own confidence score, indicating the model's certainty about each pathology's location.
            For instance, left Pleural Effusion is listed with a confidence score of 0.53, suggesting the location of Pleural Effusion to be on the left side with moderate confidence.

            When you interact with end users, remember:

            - A pathology and its lateral location (e.g., Pleural Effusion and left Pleural Effusion) are part of the same finding. The location attribute is an additional detail about where the pathology is likely found, not an indicator of a separate pathology. 
            - Synthesize the pathology detection and localization data. DO NOT TALK ABOUT THEM SEPERATELY. {"Here is a model example, 'Highly likely there is Pleural Effusion (detection confidence: 0.80), and it is possibly on the left side (localisation confidence: 0.53).'" if examples else ""}
            - Confidence scores from the pathology detection and phrase grounding tools are not directly comparable. They serve as indicators of confidence within their respective contexts of pathology detection and localisation.
            - A missing lateral location does not imply the absence of a pathology; it indicates the localisation could not be confidently determined.
            - If there is any discrepancy between the pathology detection and phrase grounding tools, detection data takes precedence as it more reliably identifies pathologies.

            It is important to factor medical knowledge and the specifics of each case, if supplied, into your responses. For example, pathologies located on both sides are called bilateral. Heart related observations are usually on the left/ middle.
            {image_context_prompt_final_part}
            """
        
        if prompt_for_chexagent_lm_output is not None:
            return system_prompt, image_context_prompt, lm_output
            
        return system_prompt, image_context_prompt



class Llama3Generation(GenerationEngine):

    def __init__(self, device = None):
        self.model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
        if device is None:
            device = select_best_gpu()
            
        self.pipeline = transformers.pipeline(
            "text-generation",
            model=self.model_id,
            model_kwargs={"torch_dtype": torch.bfloat16},
            device_map= device ,
            token=LLAMA3_INSTRUCT_ACCESS_TOKEN,
        )
  

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
            # print(format_output(output_text))
            return outputs[0]["generated_text"][len(prompt):]
        
        else:
            # setup chat loop
            chat_history = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": image_context_prompt},
            ]

            user_prompt = ""
            while user_prompt != "exit":
                prompt = self.pipeline.tokenizer.apply_chat_template(
                    chat_history, 
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
                chat_history.append({"role": "assistant", "content": output_text})
                # print(format_output(output_text))

                user_prompt = input("User: ")
                # print(f"\n{user_prompt}")
                chat_history.append({"role": "user", "content": user_prompt})
            return chat_history

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
        # print(response)
        return response
