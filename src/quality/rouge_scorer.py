from rouge_score import rouge_scorer
from rouge_score.scoring import Score
from typing import Dict, List, Sequence, Tuple
from src.models.quality.base import QualityScorerBase


class RougeQualityScorer(QualityScorerBase):

    def __init__(self, 
                rouge_metrics: List[str]= ['rouge1'],
                use_stemmer: bool=True
    ) -> None:
        self.scorer = rouge_scorer.RougeScorer(rouge_metrics, use_stemmer=use_stemmer)
        self.name = "QualityRougeScorer"

    def compute_score(self,
                    candidates: List[str], 
                    reference: str
    ) -> Sequence[Dict[str, Score]]:
        self.scores = [self.scorer.score(candidate, reference) for candidate in candidates]
        return self.scores
    
    def get_scores(self, filter_key: str=""):
        if self.scores is None:
            raise ValueError("No scores computed yet. Call compute_score first.")
        if filter_key == "":
            return self.scores
        return [score[filter_key] for score in self.scores if score.get(filter_key) is not None]