import time
from typing import Optional
from abc import ABC, abstractmethod
from multiprocessing.synchronize import Event
from multiprocessing import Process, Value
from multiprocessing.sharedctypes import Synchronized



class BaseProfiler(ABC):

    _name: str
    _time: Synchronized | float

    @abstractmethod
    def start(self) -> None:
        pass

    @abstractmethod
    def stop(self) -> None:
        pass

