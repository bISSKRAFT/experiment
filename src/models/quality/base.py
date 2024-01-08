from abc import ABC, abstractmethod
from typing import Dict, Iterable, List


class QualityScorerBase(ABC):

    scores: Iterable
    name: str 

    @abstractmethod
    def compute_score(self,
                    candidates: List[str], 
                    reference: str | List[str]
    ) -> Dict[str, float]:
        pass

    def get_scores(self):
        return self.scores
    
    def print_scores(self):
        print(self.name, "\n")
        print(self.scores)