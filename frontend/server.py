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
current_subject_index = 0

subject_to_report = {}
subject_to_image_path = {}

CHEXPERT = False
USE_STORED_REPORTS = True
DEVICE = None #"cuda:1"   
model_outputs = {}

# random.seed(42)
random.seed(50)

def initialise_models():
    global pathology_detector, phrase_grounder, l3, cheXagent_lm, cheXagent_e2e
    if USE_STORED_REPORTS:
        return
    pathology_detector = CheXagentVisionTransformerPathologyDetector(pathologies=Pathologies.CHEXPERT, device=device)
    phrase_grounder = BioVilTPhraseGrounder(detection_threshold=0.5, device = DEVICE)
    l3 = Llama3Generation(device = DEVICE)
    cheXagent_lm = CheXagentLanguageModelGeneration(pathology_detector.processor, pathology_detector.model, pathology_detector.generation_config, pathology_detector.device, pathology_detector.dtype)
    cheXagent_e2e = CheXagentEndToEndGeneration(pathology_detector.processor, pathology_detector.model, pathology_detector.generation_config, pathology_detector.device, pathology_detector.dtype)


# Read the data file into a list of dictionaries
def read_data_file(sample_random = False, no_of_scans = 50, cheXpert = CHEXPERT):

    subjects = []
    if cheXpert:
        cheXpert_small_test_path = Path("/vol/biomedic3/bglocker/ugproj2324/nns20/datasets/CheXpert/small/test")
        cheXpert_test_path = Path("/vol/biodata/data/chest_xray/CheXpert-v1.0-small/CheXpert-v1.0-small/test")

        cheXpert_subjects = Path("/vol/biomedic3/bglocker/ugproj2324/nns20/cxr-agent/evaluation_datasets/CheXpert/chexpert_test_random_written_pathologies")

        for line in cheXpert_subjects.read_text().splitlines():
            subject = line.split(",")[0]
            subjects.append(subject)
            subject_path_jpg = cheXpert_test_path / subject
            subject_to_report[subject] = line.split(",")[1:]   
            subject_to_image_path[subject] = subject_path_jpg      

        return subjects
    
    else:
        mimic_cxr_path = Path('/vol/biodata/data/chest_xray/mimic-cxr')
        mimic_cxr_jpg_path = Path('/vol/biodata/data/chest_xray/mimic-cxr-jpg')
        # subject_with_single_scan_no_prior_path = Path("/vol/biomedic3/bglocker/ugproj2324/nns20/cxr-agent/frontend/subjects_with_single_scan_no_prior.txt")
        subjects_for_eval = Path("/vol/biomedic3/bglocker/ugproj2324/nns20/cxr-agent/evaluation_datasets/MIMIC-CXR/mimic_ii_subjects_for_eval.txt")
        for line in subjects_for_eval.read_text().splitlines():
            # two cases: 1) subject with single study 2) subject with multiple studies
            if line[0] == "p":
                parts = line.split("/")
                subject = parts[0][1:] # remove the 'p' from the subject
                subjects.append(subject) 
                study = parts[1][1:]

                subject_path = mimic_cxr_path/"files"/f"p{subject[:2]}"/f"p{subject}"
                report_path = list(subject_path.glob(f"s{study}.txt"))[0]
                report = report_path.read_text()

                subject_path_jpg = mimic_cxr_jpg_path/"files"/f"p{subject[:2]}"/f"p{subject}"/f"s{study}"
                jpg_file = list(subject_path_jpg.glob('*.jpg'))[0]

            else:
                subject = line
                subjects.append(subject)

                subject_path = mimic_cxr_path/"files"/ f"p{subject[:2]}/p{subject}"
                report_path = list(subject_path.glob('*.txt'))[0]
                report = report_path.read_text()

                subject_path_jpg = mimic_cxr_jpg_path/"files"/ f"p{subject[:2]}/p{subject}"
                
                study_folder = list(subject_path_jpg.glob('*'))[0]
                jpg_file = list(study_folder.glob('*.jpg'))[0]
                
            subject_to_report[subject] = report
            subject_to_image_path[subject] = jpg_file

        return subjects


def get_model_outputs(image_path: Path):
    global model_outputs
    user_prompt = "Write a radiologist's report for the scan"
    start_time = time.time()
    
    if USE_STORED_REPORTS:
        model_outputs['chexagent'] = "This is the output of CheXagent: \n Chest X-ray: Lungs clear, no masses or infiltrates. Mediastinum normal. Abdomen: Liver, spleen, kidneys unremarkable. No bowel obstruction or free fluid. Pelvis: Bones and soft tissues normal. Extremities: No fractures, dislocations, or joint effusions. "
        model_outputs['llama3_agent'] = "This is the output of Llama3: \n Chest X-ray: Lungs clear with normal air bronchograms. No airspace disease, infiltrates, consolidation, or masses identified. Mediastinum unremarkable, aortic knob and esophagus normal caliber. Abdomen: Liver, spleen, and kidneys appear normal in size, shape, and density. No free fluid or bowel obstruction visualized. Pelvis: Bony structures demonstrate no fractures or dislocations. Urinary bladder distended normally, no calculi. Extremities: Visualized bones (e.g., femurs) demonstrate normal alignment and integrity. No joint effusions or significant osteoarthritis appreciated. "
        model_outputs['chexagent_agent'] = "This is the output of CheXagent Agent: \n"
        return model_outputs

    system_prompt, image_context_prompt, chexagent_e2e = GenerationEngine.contextualise_model(
        image_path=image_path,
        pathology_detector=pathology_detector,
        phrase_grounder=phrase_grounder,
        examples=False,
        prompt_for_chexagent_lm_output=user_prompt
    )

    model_outputs['chexagent'] = chexagent_e2e

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
    # return map in both directions
    model_name_to_id = {model: f"Model {i+1}" for i, model in enumerate(shuffled_models)}
    model_id_to_name = {f"Model {i+1}": model for i, model in enumerate(shuffled_models)}
    return model_name_to_id, model_id_to_name



### SERVER INITIALIZATION ###

initialise_models()
subjects = read_data_file()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/restart', methods=['POST'])
def restart():
    global current_subject_index
    current_subject_index = 0
    return jsonify(success=True)

@app.route('/next_image', methods=['GET'])
def next_image():
    global current_subject_index
    if current_subject_index < len(subjects):
        subject = subjects[current_subject_index]
        print(f"Current subject: {subject}")
        current_subject_index += 1

        result = {}
        global subject_to_report, subject_to_image_path

        result['report'] = subject_to_report[subject]
        
        # Convert image to base64
        full_image_path = subject_to_image_path[subject]
        print(full_image_path)
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
    model_outputs = get_model_outputs(image_path)
    
    # Randomly assign model outputs to display spaces
    model_names = list(model_outputs.keys())
    model_name_to_id, model_id_to_name = get_random_model_mapping(model_names)
    
    result = {
        'model_outputs': model_outputs,
        'model_name_to_id': model_name_to_id,
        'model_id_to_name': model_id_to_name
    }
    return jsonify(result)

@app.route('/upload_metrics', methods=['POST'])
def upload_metrics():
    data = request.get_json()
    print(data)
    return jsonify(success=True)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)