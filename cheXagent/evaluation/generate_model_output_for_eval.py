import io
import requests
import torch
from PIL import Image
from rich import print
from pathlib import Path
import os
# Setup cache for Hugging Face models
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ['HF_HOME'] = '/vol/biomedic3/bglocker/ugproj2324/nns20/CheXagent/.cache'
from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig
import random

def setup_model() -> tuple:
    device = "cuda"
    dtype = torch.float16

    processor = AutoProcessor.from_pretrained("StanfordAIMI/CheXagent-8b", trust_remote_code=True)
    generation_config = GenerationConfig.from_pretrained("StanfordAIMI/CheXagent-8b")
    print(f"{generation_config.temperature=}")
    model = AutoModelForCausalLM.from_pretrained(
        "StanfordAIMI/CheXagent-8b", torch_dtype=dtype, trust_remote_code=True
    ).to(device)

    return processor, model, device, dtype, generation_config

def generate(images, prompt, processor, model, device, dtype, generation_config):
    inputs = processor(
        images=images, text=f" USER: <s>{prompt} ASSISTANT: <s>", return_tensors="pt"
    ).to(device=device, dtype=dtype)
    output = model.generate(**inputs, generation_config=generation_config)[0]
    response = processor.tokenizer.decode(output, skip_special_tokens=True)
    return response


if __name__ == "__main__":
    model_params = setup_model() 
    # CheXpert Paths
    small_test_root = Path('/vol/biomedic3/bglocker/ugproj2324/nns20/datasets/CheXpert/small')
    small_test_csv = Path('/vol/biomedic3/bglocker/ugproj2324/nns20/datasets/CheXpert/test.csv')
    chexpert_output_directory = Path('/vol/biomedic3/bglocker/ugproj2324/nns20/cxr-agent/cheXagent/evaluation/CheXpert/chexagent')

    # VinDr Paths
    png_dset_path = Path('/vol/biodata/data/chest_xray/VinDr-CXR/1.0.0_png_512/raw/test')
    test_test_split = Path('/vol/biomedic3/bglocker/ugproj2324/nns20/datasets/VinDr-CXR/test_set_three_splits/VinDr_test_test_split.txt')
    vindr_output_directory = Path('/vol/biomedic3/bglocker/ugproj2324/nns20/cxr-agent/cheXagent/evaluation/VinDr/chexagent')


    prompts = ["What pathologies are in the image?", "What are the findings present in the image?", "What abnormalities are in the image?"]
    file_names = ["pathologies", "findings", "abnormalities"]

    #HYPERPARAMS
    repeats = 3
    temp = 0.5
    data_vindr = False

    if data_vindr:
        image_path = test_test_split 
        skip_header = False
    else:
        image_path = small_test_csv
        skip_header = True


    for prompt_index, prompt in enumerate(prompts):
        for i in range(repeats):
            responses = []
            seen_image_ids = set()
            with open(image_path) as f:
                if skip_header:
                    f.readline()
                lines = f.readlines()
                # shuffle lines
                random.shuffle(lines)
                for index, line in enumerate(lines):
                    image_id = line.split(',')[0].strip()
                    if image_id in seen_image_ids:
                        continue
                    seen_image_ids.add(image_id)

                    if data_vindr:
                        image = Image.open(png_dset_path / f"{image_id}.png").convert("RGB")
                    else:
                        image = Image.open(small_test_root/"test"/ image_id).convert("RGB") # CheXpert specific

                    response = generate([image], prompt, *model_params)
                    responses.append(f"{image_id},{response}")

                # pathologies = response.split(',')
                # for pathology in pathologies:
                #     prompt = f"Localize the {pathology}, is it on the RIGHT or the LEFT or the RIGHT AND LEFT of the given image?"
                #     response = generate([image], prompt, *model_params)
                #     responses.append(f"{image_id},{pathology},{response}")
            if data_vindr:
                file_to_write = vindr_output_directory/f'temperature_{temp}' / f'vindr_chexagent_{file_names[prompt_index]}_{i}'
            else:
                file_to_write = chexpert_output_directory/f'temperature_{temp}' / f'chexpert_chexagent_{file_names[prompt_index]}_{i}'
            with open(file_to_write, 'w') as f:
                f.write('\n'.join(responses))