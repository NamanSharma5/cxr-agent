from abc import ABC, abstractmethod
from pathlib import Path

PathologyConfidences = dict[str, float]

class PathologyDetector (ABC):

    @abstractmethod
    def detect_pathologies(self, image_path: Path) -> PathologyConfidences:
        pass

