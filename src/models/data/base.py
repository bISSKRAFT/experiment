from abc import ABC, abstractmethod

class BaseDatasetMixin(ABC):

    @abstractmethod
    def get_data(self):
        """
        Returns the preprocessd data of the dataset.
        """
        pass