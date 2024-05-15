import os
os.environ['HF_HOME'] = '/vol/biomedic3/bglocker/ugproj2324/nns20/cxr-agent/.hf_cache' ## THIS HAS TO BE BEFORE YOU IMPORT TRANSFORMERS

from health_multimodal.common.visualization import plot_phrase_grounding_similarity_map
from health_multimodal.text import get_bert_inference
from health_multimodal.text.utils import BertEncoderType
from health_multimodal.image import get_image_inference
from health_multimodal.image.utils import ImageModelType
from health_multimodal.vlp import ImageTextInferenceEngine

from agent_utils import select_best_gpu
from abc import ABC, abstractmethod
from typing import Union
from collections import defaultdict

PathologyLocationConfidences = dict[str, float]

DECIMALS = 2

class PhraseGrounder(ABC):

    @abstractmethod
    def get_similiarity_map(self, phrase: str, image_path: str):
        pass

    @abstractmethod 
    def get_pathology_lateral_position(self, patholgy: str, image_path: str) -> str:
        pass


class BioVilTPhraseGrounder(PhraseGrounder):

    def __init__(self, detection_threshold = 0.25, top_n_pixels = 25, device = None):
        self.detection_threshold = detection_threshold
        self.top_n_pixels = top_n_pixels

        self.text_inference = get_bert_inference(BertEncoderType.BIOVIL_T_BERT)
        self.image_inference = get_image_inference(ImageModelType.BIOVIL_T)  
        self.image_text_inference_engine = ImageTextInferenceEngine(image_inference_engine=self.image_inference, text_inference_engine=self.text_inference)
        
        if device is None:
            self.device = select_best_gpu()
        else:
            self.device = device
            
        self.image_text_inference_engine.to(self.device)

    def get_similiarity_map(self, image_path: str, phrase: str):
        return self.image_text_inference_engine.get_similarity_map_from_raw_data(
            image_path=image_path,
            query_text=phrase,
            interpolation="bilinear"
        )
    
    def get_top_values(self, similarity_map, return_mean = False):
        top_values = []
        for i in range(similarity_map.shape[0]):
            for j in range(similarity_map.shape[1]):
                if similarity_map[i, j] > self.detection_threshold:
                    top_values.append((i, j, similarity_map[i, j]))

        top_values = sorted(top_values, key = lambda x: x[2], reverse = True)

        if return_mean:
            if len(top_values) == 0:
                return [], 0
            return top_values[:self.top_n_pixels], sum([x[2] for x in top_values]) / len(top_values)
        
        return top_values[:self.top_n_pixels]

    def get_pathology_lateral_position(self, image_path: str, pathologies: Union[str,list[str]]) -> PathologyLocationConfidences:
        if isinstance(pathologies, str):
            pathologies = [pathologies]

        locations = ["left", "right"]

        pathologyLocationConfidences = defaultdict(list)

        for pathology in pathologies:
            for location in locations:
                phrase = f"{location} {pathology}"
                similarity_map = self.get_similiarity_map(image_path, phrase)
                _ , mean_activation = self.get_top_values(similarity_map, return_mean = True)

                if mean_activation > 0:
                    pathologyLocationConfidences[pathology].append((location,round(mean_activation,DECIMALS)))

        return dict(pathologyLocationConfidences)
            
            
        

