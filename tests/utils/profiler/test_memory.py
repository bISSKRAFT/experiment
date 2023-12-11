import time
from typing import List
import torch
from src.utils.profiler.memory_profiler import MemoryProfilerCallback, MemoryProfilerProcess



def test_mem_profiler_process() -> None:
    torch.multiprocessing.set_start_method('spawn')
    crtl_used_vram = torch.cuda.mem_get_info(0)[1] - torch.cuda.mem_get_info(0)[0]
    profiler = MemoryProfilerProcess("test", 0, crtl_used_vram)
    profiler.start()
    # simulate inference 
    time.sleep(5)
    profiler.stop()
    max_vram = profiler.get_max_vram
    consumed_time = profiler.get_time
    print(max_vram, consumed_time)
    assert max_vram < 500
    assert consumed_time > 0.0

def test_properites_MemoryProfilerCallback():
    profiler = MemoryProfilerCallback("test")
    profiler._time = 120.0
    assert profiler.inference_time == 120.0

def test_while_loop() -> None:
    device = torch.cuda.current_device()
    print(torch.cuda.memory_stats(device))
