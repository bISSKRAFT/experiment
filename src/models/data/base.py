from abc import ABC, abstractmethod

class BaseDatasetMixin(ABC):

    @abstractmethod
    def get_data(self):
        """
        Returns the preprocessd data of the dataset.
        """
        pass

    @abstractmethod
    def caluclate_score(self, targets, predictions):
        """
        Returns the score of the model on the dataset.
        """
        pass