import nvidia_smi
from src.utils.profiler.memory_profiler import MemoryProfilerCallback





def test_placeholder():
    nvidia_smi.nvmlInit()
    handle = nvidia_smi.nvmlDeviceGetHandleByIndex(0)
    for process in nvidia_smi.nvmlDeviceGetComputeRunningProcesses(handle):
        print(nvidia_smi.nvmlSystemGetProcessName(process.pid))
        print("MEM: ",process.usedGpuMemory / 1024 / 1024, "\n")
    nvidia_smi.nvmlShutdown()