from abc import ABC
import time
from abc import ABC, abstractmethod
from multiprocessing.synchronize import Event
from multiprocessing import Process, Value
from multiprocessing.sharedctypes import Synchronized
import time
from typing import Optional
from src.models.profiler.base import BaseProfiler


class BaseMemoryProfiler(BaseProfiler, ABC):

    _max_vram: float | Synchronized
    _process: Optional[Process]
    _stop_event: Event

    @property
    def get_max_vram(self) -> float:
        if isinstance(self._max_vram, Synchronized):
            return self._max_vram.value / 1024 / 1024
        return self._max_vram / 1024 / 1024
    
    @property
    def get_time(self) -> float:
        if isinstance(self._time, Synchronized):
            return self._time.value
        return self._time
    
    @abstractmethod
    def _cleanup(self) -> None:
        pass

    @abstractmethod
    def _memory_monitor(self) -> None:
        pass
    
    # async functions

    @abstractmethod
    def _create_process(self, 
                       stop_event: Event
                       ) -> Process:
        pass

    def start(self) -> None:
        self._time = Value("d", time.perf_counter())
        if self._stop_event is None:
            raise ValueError("stop_event must be set before starting process")
        self._process = self._create_process(self._stop_event)
        self._process.start()

    def stop(self) -> None:
        self._stop_event.set()
        assert self._process is not None
        self._process.join()
        if not isinstance(self._time, Synchronized):
            raise ValueError("time must be Synchronized")
        self._time = Value("d", time.perf_counter() - self._time.value)
        self._cleanup()