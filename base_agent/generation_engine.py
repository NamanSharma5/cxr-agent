from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

class GenerationEngine(ABC):
    @abstractmethod
    def generate_report(self, image_path: Path, prompt: Optional[str], output_dir: Optional[str]) -> str:
        pass


class Llama3Generation(GenerationEngine):

    
    def generate_report(self, image_path: Path, prompt: Optional[str], output_dir: Optional[str]) -> str:
    pass