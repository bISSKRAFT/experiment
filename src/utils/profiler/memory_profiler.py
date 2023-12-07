import asyncio
import time
import torch
from multiprocessing.sharedctypes import Synchronized
from multiprocessing import Value, Event, Process
from multiprocessing.synchronize import Event as EventClass
from typing import Optional, Dict, List
import time

from src.models.profiler.memory import BaseMemoryProfiler

_MEMORY_STATS = {
    'allocation.all.allocated': 'alloc_requests',
    'allocation.all.freed': 'free_requests',
    'allocated_bytes.all.allocated': 'allocated_mem',
    'active_bytes.all.current': 'active_mem',
    'inactive_split_bytes.all.current': 'inactive_mem',
    'reserved_bytes.all.current': 'reserved_mem',
    'num_alloc_retries': 'alloc_retries',
}

class MemoryProfilerCallback():

    def __init__(self, name: str):
        self._name = name
        self._time: float = 0.0
        self._max_vram = 0.0

    @property
    def inference_time(self) -> float:
        return self._time

    def memory_report(self) -> Dict[str, float]:
        # if device.type != "cuda":
        #     raise ValueError("device is not cuda")
        memory_stats = torch.cuda.memory_stats()

        memory_report = {
            name: memory_stats[torch_name] for (torch_name, name) in _MEMORY_STATS.items() if torch_name in memory_stats
        }

        return memory_report

    def organize_memory_report(self, memory_report: List[dict]) -> Dict[str, list]:
        return {key: [dic[key] for dic in memory_report] for key in memory_report[0]}
    


class MemoryProfilerProcess(BaseMemoryProfiler):

    def __init__(self, name: str, gpu_handle: int, offset_vram: float = 0.0):
        self._name = name
        self._time: Synchronized = Value("d", 0.0)
        self._total_ram: Synchronized = Value("d",0.0)
        self._max_ram: Synchronized = Value("d",0.0)
        self._total_vram: Synchronized = Value("d",0.0)
        self._max_vram: Synchronized = Value("d",0.0)
        self._offset_vram: Synchronized = Value("d",offset_vram)
        self._gpu_handle: Synchronized = Value("i",gpu_handle)

        self._stop_event = Event()
        self._process: Optional[Process] = None

    def _create_process(self, 
                       stop_event: EventClass, 
                       ) -> Process:
        process = Process(
            target=self._memory_monitor, 
            args=(
                stop_event, 
                self._max_vram, 
                self._gpu_handle,
                )
            )
        return process
    
    def get_statistics(self):
        return self._max_vram.value / 1024 / 1024, self._total_vram.value / 1024 / 1024, self._time.value

    def _memory_monitor(self,
            stop_event: EventClass,
            max_vram: Synchronized,
            gpu_handle: Synchronized,
        )-> None:

        max_vram_value = 0.0
        
        while not stop_event.is_set():
            vram_tpl = torch.cuda.mem_get_info(gpu_handle.value)
            max_vram_value = max(max_vram_value, vram_tpl[1] - vram_tpl[0]) 
            time.sleep(0.2)

        if max_vram_value == 0:
            raise ValueError("max_vram_value is 0")
        max_vram.value = max_vram_value

    def _cleanup(self) -> None:
        self._total_vram.value = torch.cuda.mem_get_info(0)[1]
        self._max_vram.value = self._max_vram.value - self._offset_vram.value

