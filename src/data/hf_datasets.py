from typing import List, Tuple
from src.models.data.base import BaseDatasetMixin
from datasets import load_dataset


class HellaSwagDataset(BaseDatasetMixin):

    def __init__(self):
        self.dataset = load_dataset(
            "Rowan/hellaswag",
            split="train",
            streaming=False)

    def get_data(self) -> Tuple[List, List, List]:
        context = list(self.dataset["ctx"])  # Convert dataset to list
        candidates = list(self.dataset["endings"])  # Convert dataset to list
        labels = list(self.dataset["label"])  # Convert dataset to list
        assert len(context) == len(candidates) == len(labels)
        return context, candidates, labels
    
class ARCDataset(BaseDatasetMixin):
    
    def __init__(self):
        self.dataset = load_dataset(
            "allenai/ai2_arc",
            "ARC-Challenge",
            streaming=False)

    def get_data(self) -> Tuple[List, List, List]:
        context = list(self.dataset["question"])  # Convert dataset to list
        candidates = list(self.dataset["choices"]["text"])  # Convert dataset to list
        labels = list(self.dataset["answerKey"])  # Convert dataset to list
        assert len(context) == len(candidates) == len(labels)
        return context, candidates, labels
