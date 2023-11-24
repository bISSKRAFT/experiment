from abc import ABC, abstractmethod
from multiprocessing.synchronize import Event
from multiprocessing import Process, Value
from multiprocessing.sharedctypes import Synchronized
import time
from typing import Optional

class BaseProfiler(ABC):

    _name: str
    _time: Synchronized | float

    @abstractmethod
    def start(self) -> None:
        pass

    @abstractmethod
    def stop(self) -> None:
        pass

# class BaseAsyncProfiler(BaseProfiler,ABC):

#     _time: Synchronized
#     _process: Optional[Process]
#     _stop_event: Event

#     @abstractmethod
#     def create_process(self, 
#                        stop_event: Event
#                        ) -> Process:
#         pass

#     def start(self) -> None:
#         self._time = Value("d", time.perf_counter())
#         if self._stop_event is None:
#             raise ValueError("stop_event must be set before starting process")
#         self._process = self.create_process(self._stop_event)
#         self._process.start()

#     def stop(self) -> None:
#         self._stop_event.set()
#         assert self._process is not None
#         self._process.join()
#         self._time = Value("d", time.perf_counter() - self._time.value)
