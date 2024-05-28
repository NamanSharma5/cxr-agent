from flask import Flask, render_template, jsonify, request
from livereload import Server, shell
from pathlib import Path
import os
import base64
import random
import json
import sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import pickle
import csv
from datetime import datetime

# Get the current script's directory
current_dir = Path(__file__).resolve().parent

# Add the base_agent directory to sys.path
base_agent_dir = current_dir.parent / 'base_agent'
sys.path.insert(0, str(base_agent_dir))

from pathology_sets import Pathologies
from pathology_detector import CheXagentVisionTransformerPathologyDetector
from phrase_grounder import BioVilTPhraseGrounder
from generation_engine import GenerationEngine, Llama3Generation, CheXagentLanguageModelGeneration, GeminiFlashGeneration

app = Flask(__name__)

# CONFIGURATION
USER_PROMPT = "Generate the findings section of the radiology report for the scan"
USER_PROMPT = "What are the findings?"

PATHOLOGY_DETECTION_THRESHOLD = 0.4
PHRASE_GROUNDING_THRESHOLD = 0.2
IGNORE_PATHOLOGIES = {"Support Devices"}
DO_NOT_LOCALISE = {"Cardiomegaly"}

CHEXPERT = True
SHUFFLE = True

DEVICE = None #"cuda:1"  
USE_STORED_REPORTS = True
FILE_NAME = "findings_section_of_report" #"findings" #"radiology_report" = too verbose for L3

SAVE_RESULTS = True
stored_responses_path = Path("/vol/biomedic3/bglocker/ugproj2324/nns20/cxr-agent/frontend/stored_responses")

# Directory prefix for the images
current_subject_index = 0

subject_to_report = {}
subject_to_image_path = {}
model_outputs = {}
file_path = None

random.seed(5)

def initialise_models():
    global pathology_detector, phrase_grounder, l3, cheXagent_lm, gemini, image_path_to_model_outputs
    if USE_STORED_REPORTS:        
        if CHEXPERT:
            chexpert_findings_path = stored_responses_path / "CheXpert" / f"{FILE_NAME}.pkl"
            image_path_to_model_outputs = pickle.load(open(chexpert_findings_path, "rb"))
        else:
            mimic_findings_path = stored_responses_path / "MIMIC-CXR" / f"{FILE_NAME}.pkl"
            image_path_to_model_outputs = pickle.load(open(mimic_findings_path, "rb"))
        return
    pathology_detector = CheXagentVisionTransformerPathologyDetector(pathologies=Pathologies.CHEXPERT, device=DEVICE)
    phrase_grounder = BioVilTPhraseGrounder(detection_threshold=PHRASE_GROUNDING_THRESHOLD, device = DEVICE)
    l3 = Llama3Generation(device = DEVICE)
    cheXagent_lm = CheXagentLanguageModelGeneration(pathology_detector.processor, pathology_detector.model, pathology_detector.generation_config, pathology_detector.device, pathology_detector.dtype)
    gemini = GeminiFlashGeneration()

# Read the data file into a list of dictionaries
def read_data_file(shuffle = SHUFFLE, no_of_scans = 50, cheXpert = CHEXPERT):

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

    if shuffle:
        random.shuffle(subjects)

    return subjects


def get_model_outputs(image_path: Path):
    global model_outputs, image_path_to_model_outputs
    start_time = time.time()
    
    if USE_STORED_REPORTS:
        model_outputs = image_path_to_model_outputs[image_path]
        return model_outputs

    pathology_confidences, localised_pathologies, chexagent_e2e = GenerationEngine.detect_and_localise_pathologies(
        image_path=image_path,
        pathology_detector=pathology_detector,
        phrase_grounder=phrase_grounder,
        pathology_detection_threshold = PATHOLOGY_DETECTION_THRESHOLD,
        ignore_pathologies=IGNORE_PATHOLOGIES,
        do_not_localise=DO_NOT_LOCALISE,
        prompt_for_chexagent_lm_output= USER_PROMPT,
    )
    
    print(f"Pathology confidences: {pathology_confidences}")
    print(f"Localised pathologies: {localised_pathologies}")
    model_outputs['chexagent'] = chexagent_e2e

    gemini_system_prompt, gemini_image_context_prompt = gemini.generate_prompts(pathology_confidences, localised_pathologies)
    model_outputs['gemini_agent'] = gemini.generate_model_output(gemini_system_prompt, gemini_image_context_prompt, user_prompt=USER_PROMPT)

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

def create_file():
    global file_path
    directory = '/vol/biomedic3/bglocker/ugproj2324/nns20/cxr-agent/frontend/evaluation_metrics'
    os.makedirs(directory, exist_ok=True)

    if CHEXPERT:
        file_path = os.path.join(directory, f"chexpert_{datetime.now().strftime('%m%d_%H%M')}_{FILE_NAME}.csv")
    else:
        file_path = os.path.join(directory, f"mimic_{datetime.now().strftime('%m%d_%H%M')}_{FILE_NAME}.csv")    
    
    # Define the columns
    headers = ['subject', 'abnormal',
               'chexagent_rank', 'chexagent_agent_rank', 'llama3_agent_rank', 'gemini_agent_rank',
               'chexagent_rubric', 'chexagent_agent_rubric', 'llama3_agent_rubric', 'gemini_agent_rubric',
               'chexagent_brevity', 'chexagent_agent_brevity', 'llama3_agent_brevity', 'gemini_agent_brevity',
               'chexagent_accuracy', 'chexagent_agent_accuracy', 'llama3_agent_accuracy', 'gemini_agent_accuracy',
            #    'chexagent_missed_pathology', 'chexagent_agent_missed_pathology', 'llama3_agent_missed_pathology', 'gemini_agent_missed_pathology']
               'chexagent_dangerous', 'chexagent_agent_dangerous', 'llama3_agent_dangerous', 'gemini_agent_dangerous']
    
    # Create the file with headers
    with open(file_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()


### SERVER INITIALIZATION ###

initialise_models()
subjects = read_data_file()
if SAVE_RESULTS:
    create_file()

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

    if SAVE_RESULTS:
        global file_path
        
        # Extract metadata
        subject = data.get('metadata', {}).get('subject', '')
        abnormal = data.get('metadata', {}).get('abnormal', False)
        
        # Create a row with default None values
        row = { 
                'subject': subject,
                'abnormal': abnormal,
                'chexagent_rank': None, 'chexagent_agent_rank': None, 'llama3_agent_rank': None, 'gemini_agent_rank': None,
                'chexagent_rubric': None, 'chexagent_agent_rubric': None, 'llama3_agent_rubric': None, 'gemini_agent_rubric': None,
                'chexagent_brevity': None, 'chexagent_agent_brevity': None, 'llama3_agent_brevity': None, 'gemini_agent_brevity': None,
                'chexagent_accuracy': None, 'chexagent_agent_accuracy': None, 'llama3_agent_accuracy': None, 'gemini_agent_accuracy': None,
                # 'chexagent_missed_pathology': None, 'chexagent_agent_missed_pathology': None, 'llama3_agent_missed_pathology': None, 'gemini_agent_missed_pathology': None
                'chexagent_dangerous': None, 'chexagent_agent_dangerous': None, 'llama3_agent_dangerous': None, 'gemini_agent_dangerous': None
            }

        # Fill in the row with actual data
        for model_name, metrics in data.items():
            if model_name == 'metadata':
                continue
            for metric, value in metrics.items():
                row[f'{model_name}_{metric}'] = value

        # Write the row to the CSV file
        with open(file_path, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=row.keys())
            writer.writerow(row)

    print(data)
    return jsonify(success=True)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=4000, debug=True, use_reloader=True)

