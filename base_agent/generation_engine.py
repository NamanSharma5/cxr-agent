import os
os.environ["HF_HOME"] = "/vol/biomedic3/bglocker/ugproj2324/nns20/llama3/.8b-instruct-cached"

import transformers
import torch

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional
from my_secrets import LLAMA3_INSTRUCT_ACCESS_TOKEN
from agent_utils import select_best_gpu

from pathology_detector import PathologyDetector, CheXagentVisionTransformerPathologyDetector

class GenerationEngine(ABC):
    @abstractmethod
    def generate_report(self, image_path: Path, prompt: Optional[str], output_dir: Optional[str]) -> str:
        pass


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


    def generate_report(self, image_path: Path, prompt: Optional[str], pathology_detector = None) -> str:
        
        if pathology_detector is not None:
            pathology_confidences = pathology_detector.detect_pathologies(image_path, threshold = 0.5)
        else:
            pathology_confidences = { 'Consolidation': 0.21995275,'Pulmonary fibrosis': 0.49335942, 'No finding': 0.1746179}

        system_prompt = """You are a helpful assistant, specialising in radiology and interpreting Chest X-rays. Please answer concisely and in a professional manner."""

        user_prompt = f"""Using specialised pathology detection tools,
        you are given the following pathology detection results for a chest X-ray:
        {pathology_confidences}

        Please note the closer the value to 1, the more likely the pathology is present in the image. 
        Write up a findings section based on these observations"""


        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt },
        ]

        prompt = self.pipeline.tokenizer.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True
        )

        terminators = [
            self.pipeline.tokenizer.eos_token_id,
            self.pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]

        outputs = self.pipeline(
            prompt,
            max_new_tokens=256,
            eos_token_id=terminators,
            do_sample=True,
            temperature=0.6,
            top_p=0.9,
        )

        print(outputs[0]["generated_text"][len(prompt):])
        return outputs[0]["generated_text"][len(prompt):]


# if __name__ == "__main__":
#     l3 = Llama3Generation()
#     pathology_detector = CheXagentVisionTransformerPathologyDetector()

#     chexpert_test_csv_path = Path("/vol/biodata/data/chest_xray/CheXpert-v1.0-small/CheXpert-v1.0-small/test.csv")
#     chexpert_test_path = Path("/vol/biomedic3/bglocker/ugproj2324/nns20/datasets/CheXpert/small/")

#     with open(chexpert_test_csv_path, 'r') as f:
#         lines = f.readlines()
#         header = lines[0].split(",")[1:]
#         # print(header)
#         for i, line in enumerate(lines[1:]):
#             if i % 1000 == 0:
#                 print(f"Collecting image {i}")

#             image_path = line.split(",")[0]
#             image_path = chexpert_test_path / image_path

#             l3.generate_report(image_path, prompt = None, pathology_detector=pathology_detector)
#             break
