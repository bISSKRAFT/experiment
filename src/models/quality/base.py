from abc import ABC, abstractmethod
from typing import Dict, List


class QualityScorerBase(ABC):

    scores = None

    @abstractmethod
    def compute_score(self,
                    candidates: List[str], 
                    references: List[str]
    ) -> Dict[str, float]:
        pass

    def get_scores(self):
        return self.scores