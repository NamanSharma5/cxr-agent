from flask import Flask, render_template, jsonify, request
from pathlib import Path
import os
import base64
import random
import json
import sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

# Get the current script's directory
current_dir = Path(__file__).resolve().parent

# Add the base_agent directory to sys.path
base_agent_dir = current_dir.parent / 'base_agent'
sys.path.insert(0, str(base_agent_dir))

from pathology_sets import Pathologies
from pathology_detector import CheXagentVisionTransformerPathologyDetector
from phrase_grounder import BioVilTPhraseGrounder
from generation_engine import GenerationEngine, Llama3Generation, CheXagentLanguageModelGeneration, CheXagentEndToEndGeneration

app = Flask(__name__)

# Directory prefix for the images
IMAGE_DIR_PREFIX = Path("/vol/biodata/data/chest_xray/mimic-cxr-jpg/files/")
# To keep track of the current line
current_line_index = 0

subject_to_report = {}
subject_to_image_path = {}

WITHOUT_MODELS = False
model_outputs = {}

random.seed(42)

def initialise_models(without_models=WITHOUT_MODELS):
    global pathology_detector, phrase_grounder, l3, cheXagent_lm, cheXagent_e2e
    if without_models:
        return
    device = "cuda:1"
    pathology_detector = CheXagentVisionTransformerPathologyDetector(pathologies=Pathologies.CHEXPERT, device=device)
    phrase_grounder = BioVilTPhraseGrounder(detection_threshold=0.5, device = device)
    l3 = Llama3Generation(device = device)
    cheXagent_lm = CheXagentLanguageModelGeneration(pathology_detector.processor, pathology_detector.model, pathology_detector.generation_config, pathology_detector.device, pathology_detector.dtype)
    cheXagent_e2e = CheXagentEndToEndGeneration(pathology_detector.processor, pathology_detector.model, pathology_detector.generation_config, pathology_detector.device, pathology_detector.dtype)


# Read the data file into a list of dictionaries
def read_data_file(sample_random = True, no_of_scans = 50):

    mimic_cxr_path = Path('/vol/biodata/data/chest_xray/mimic-cxr')
    mimic_cxr_jpg_path = Path('/vol/biodata/data/chest_xray/mimic-cxr-jpg')
    subject_with_single_scan_no_prior_path = Path("/vol/biomedic3/bglocker/ugproj2324/nns20/cxr-agent/frontend/subjects_with_single_scan_no_prior.txt")
    
    subjects = subject_with_single_scan_no_prior_path.read_text().splitlines()

    if sample_random:
        subjects = random.sample(subjects, no_of_scans)
    else:
        subjects = subjects[:no_of_scans]

    for subject in subjects:
            subject_path = mimic_cxr_path/"files"/ f"p{subject[:2]}/p{subject}"
            report_path = list(subject_path.glob('*.txt'))[0]
            report = report_path.read_text()

            subject_path_jpg = mimic_cxr_jpg_path/"files"/ f"p{subject[:2]}/p{subject}"
            
            study_folder = list(subject_path_jpg.glob('*'))[0]
            jpg_file = list(study_folder.glob('*.jpg'))[0]
            
            subject_to_report[subject] = report
            subject_to_image_path[subject] = jpg_file

    return subjects


def get_model_outputs(image_path: Path, without_models= WITHOUT_MODELS):
    global model_outputs
    user_prompt = "Write a radiologist's report for the scan"
    start_time = time.time()
    
    if without_models:
        model_outputs['chexagent'] = "This is the output of CheXagent: \n Chest X-ray: Lungs clear, no masses or infiltrates. Mediastinum normal. Abdomen: Liver, spleen, kidneys unremarkable. No bowel obstruction or free fluid. Pelvis: Bones and soft tissues normal. Extremities: No fractures, dislocations, or joint effusions. "
        model_outputs['llama3_agent'] = "This is the output of Llama3: \n Chest X-ray: Lungs clear with normal air bronchograms. No airspace disease, infiltrates, consolidation, or masses identified. Mediastinum unremarkable, aortic knob and esophagus normal caliber. Abdomen: Liver, spleen, and kidneys appear normal in size, shape, and density. No free fluid or bowel obstruction visualized. Pelvis: Bony structures demonstrate no fractures or dislocations. Urinary bladder distended normally, no calculi. Extremities: Visualized bones (e.g., femurs) demonstrate normal alignment and integrity. No joint effusions or significant osteoarthritis appreciated. "
        model_outputs['chexagent_agent'] = "This is the output of CheXagent Agent: \n"
        return model_outputs

    # Define tasks to run in parallel
    def run_contextualise_model():
        return GenerationEngine.contextualise_model(
            image_path=image_path,
            pathology_detector=pathology_detector,
            phrase_grounder=phrase_grounder,
            examples=False,
            prompt_for_chexagent_lm_output=user_prompt
        )

        # Wait for GenerationEngine.contextualise_model to finish
    system_prompt, image_context_prompt, chexagent_e2e = GenerationEngine.contextualise_model(
        image_path=image_path,
        pathology_detector=pathology_detector,
        phrase_grounder=phrase_grounder,
        examples=False,
        prompt_for_chexagent_lm_output=user_prompt
    )

    model_outputs['chexagent'] = chexagent_e2e

    # def run_chexagent_e2e(system_prompt, image_context_prompt):
    #     return cheXagent_e2e.generate_model_output(system_prompt, image_context_prompt, user_prompt=user_prompt, image_path=image_path)
    
    def run_chexagent_lm(system_prompt, image_context_prompt):
        return cheXagent_lm.generate_model_output(system_prompt, image_context_prompt, user_prompt=user_prompt)

    def run_llama3_agent(system_prompt, image_context_prompt):
        return l3.generate_model_output(system_prompt, image_context_prompt, user_prompt=user_prompt)

    with ThreadPoolExecutor() as executor:       

        # Run cheXagent_lm and l3 in parallel
        future_chexagent_lm = executor.submit(run_chexagent_lm, system_prompt, image_context_prompt)
        future_llama3_agent = executor.submit(run_llama3_agent, system_prompt, image_context_prompt)

        for future in as_completed([future_chexagent_lm, future_llama3_agent]):
            if future == future_chexagent_lm:
                model_outputs['chexagent_agent'] = future.result()
            else:
                model_outputs['llama3_agent'] = future.result()
    
    print(f"Time taken: {time.time() - start_time}")
    return model_outputs


def get_random_model_mapping(models):
    shuffled_models = models[:]
    random.shuffle(shuffled_models)
    return {model: f"Model {i+1}" for i, model in enumerate(shuffled_models)}

### SERVER INITIALIZATION ###

initialise_models()
subjects = read_data_file()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/restart', methods=['POST'])
def restart():
    global current_line_index
    current_line_index = 0
    return jsonify(success=True)

@app.route('/next_image', methods=['GET'])
def next_image():
    global current_line_index
    if current_line_index < len(subjects):
        subject = subjects[current_line_index]
        current_line_index += 1

        result = {}
        global subject_to_report, subject_to_image_path

        result['report'] = subject_to_report[subject]
        
        # Convert image to base64
        full_image_path = subject_to_image_path[subject]
        with open(full_image_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
        result['image_data'] = encoded_string

        # Return image and report only
        result['subject'] = subject

    else:
        result = {'image_data': '', 'report': 'No more data available.', 'subject': ''}
    return jsonify(result)

@app.route('/get_model_outputs', methods=['POST'])
def get_model_outputs_route():
    data = request.get_json()
    subject = data['subject']
    image_path = subject_to_image_path[subject]

    # Fetch model outputs
    model_outputs = get_model_outputs(image_path, without_models=True)
    
    # Randomly assign model outputs to display spaces
    model_names = list(model_outputs.keys())
    model_mapping = get_random_model_mapping(model_names)
    
    result = {
        'model_outputs': model_outputs,
        'model_mapping': model_mapping
    }
    return jsonify(result)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)