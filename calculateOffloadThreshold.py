import os
import sys
import csv
from enum import Enum
from texttable import Texttable

#### Supported Kernel enum
class kernels(Enum):
    SGEMM = 1
    DGEMM = 2
    SGEMV = 3
    DGEMV = 4
    NONE = 0

#### Function to convert from string to kernels ENUM
def strToKernel(kernel:str) -> kernels:
    if kernel == "sgemm":
        return kernels.SGEMM
    if kernel == "dgemm":
        return kernels.DGEMM
    if kernel == "sgemv":
        return kernels.SGEMV
    if kernel == "dgemv":
        return kernels.DGEMV
    return kernels.NONE

#### Supported Kernels table headers
gemmHeaders = ["Device", "M", "N", "K", "Total Prob. Size (KiB)", "GFLOP/s", "CPU GFLOP/s"]
gemvHeaders = ["Device", "M", "N", "Total Prob. Size (KiB)", "GFLOP/s", "CPU GFLOP/s"]

#### Offload threshold class
class offloadThreshold:
    def __init__(self, kernel:kernels) -> None:
        self.cpuGflops = 0.0
        self.gpuGflops = 0.0
        self.totalKib = 0.0
        self.M = 0
        self.N = 0
        self.K = 0

#### Function to print table to console
def printResults(once:offloadThreshold, always:offloadThreshold, unified:offloadThreshold, kernel:kernels):
    rows = []
    if(kernel == kernels.DGEMM or kernel == kernels.SGEMM):
        rows.append(gemmHeaders)
        rows.append(["GPU (Offload Once)", once.M, once.N, once.K, round(once.totalKib, 3), round(once.gpuGflops, 3), round(once.cpuGflops, 3)])
        rows.append(["GPU (Offload Always)", always.M, always.N, always.K, round(always.totalKib, 3), round(always.gpuGflops, 3), round(always.cpuGflops, 3)])
        rows.append(["GPU (Unified Memory)", unified.M, unified.N, unified.K, round(unified.totalKib, 3), round(unified.gpuGflops, 3), round(unified.cpuGflops, 3)])
    elif(kernel == kernels.DGEMV or kernel == kernels.SGEMV):
        rows.append(gemvHeaders)
        rows.append(["GPU (Offload Once)", once.M, once.N, round(once.totalKib, 3), round(once.gpuGflops, 3), round(once.cpuGflops, 3)])
        rows.append(["GPU (Offload Always)", always.M, always.N, round(always.totalKib, 3), round(always.gpuGflops, 3), round(always.cpuGflops, 3)])
        rows.append(["GPU (Unified Memory)", unified.M, unified.N, round(unified.totalKib, 3), round(unified.gpuGflops, 3), round(unified.cpuGflops, 3)])
    else:
        exit(1)
    table = Texttable()
    table.add_rows(rows)
    print(table.draw())
    


#########################################################################################
cpuCSV_path = ""
gpuCSV_path = ""
# Define offload threshold classes
gpuOnce = offloadThreshold(kernels.NONE)
gpuAlways = offloadThreshold(kernels.NONE)
gpuUnified = offloadThreshold(kernels.NONE)

# Check both filenames have been passed in
if(len(sys.argv) == 3):
    cpuCSV_path = sys.argv[1]
    gpuCSV_path = sys.argv[2]
else:
    print("ERROR - Must provide the CPU and GPU CSV file paths:")
    print("\t\tpython calculateOffloadThreshold.py path/to/cpu.csv path/to/gpu.csv")
    exit(1)


# Check both CSV files exist
if not os.path.isfile(cpuCSV_path):
    print("ERROR - CPU CSV file \"{}\" cannot be found.".format(cpuCSV_path))
    exit(1)
if not os.path.isfile(gpuCSV_path):
    print("ERROR - GPU CSV file \"{}\" cannot be found.".format(gpuCSV_path))
    exit(1)


# Open files
cpuFile = open(cpuCSV_path, 'r')
gpuFile = open(gpuCSV_path, 'r')

### For evey 1 CPU entry, there should be 3 GPU entries
### CSV header format ==== Device,Kernel,M,N,K,Total Problem Size (KiB),Iterations,Total Seconds,GFLOP/s 
cpuLines = cpuFile.readlines()
gpuLines = gpuFile.readlines()

# Remove first line of file - header names
cpuLines.pop(0)
gpuLines.pop(0)

# Check that there are 3x as many GPU lines as CPU lines
if(len(cpuLines) * 3 != len(gpuLines)):
    print("ERROR - Number of GPU records does not match number of CPU records (There should be 3 GPU entries for each CPU entry).")
    exit(1)

# Go through all entries and find offload threshold
kernel = ""
prevGpuOgflops = 0.0
prevGpuAgflops = 0.0
prevGpuUgflops = 0.0
for cpu in cpuLines:
    cpu = cpu.split(',')
    gpuO = gpuLines.pop(0).split(',')
    gpuA = gpuLines.pop(0).split(',')
    gpuU = gpuLines.pop(0).split(',')
    # Make sure entries are correct
    if(cpu[0] != "cpu"):
        print("ERROR - Non-CPU entry in CPU CSV file.")
        exit(1)
    if(gpuO[0] != "gpu_offloadOnce" or 
       gpuA[0] != "gpu_offloadAlways" or 
       gpuU[0] != "gpu_unified"):
        print("ERROR - Non-GPU entry in GPU CSV file.")
        exit(1)
    # Make sure kernels are the same
    if(cpu[1] != gpuO[1] or cpu[1] != gpuA[1] or cpu[1] != gpuU[1]):
        print("ERROR - kernel mismatch.")
        exit(1)
    else:
        kernel = cpu[1]
    # Make sure all M values match
    if(cpu[2] != gpuO[2] or cpu[2] != gpuA[2] or cpu[2] != gpuU[2]):
        print("ERROR - values of M mismatch.")
        exit(1)
    # Make sure all N values match
    if(cpu[3] != gpuO[3] or cpu[3] != gpuA[3] or cpu[3] != gpuU[3]):
        print("ERROR - values of N mismatch.")
        exit(1)
    # Make sure all K values match
    if(cpu[4] != gpuO[4] or cpu[4] != gpuA[4] or cpu[4] != gpuU[4]):
        print("ERROR - values of K mismatch.")
        exit(1)
    # Make sure iteration count match
    if(cpu[6] != gpuO[6] or cpu[6] != gpuA[6] or cpu[6] != gpuU[6]):
        print("ERROR - number of iterations mismatch.")
        exit(1)

    # Check if offload structures should be reset (CPU.gflops >= GPU.gflops)
    if(gpuOnce.M != 0 and float(cpu[8]) >= float(gpuO[8])):
        # Do check to see if this is a momentary drop that we should ignore
        if (prevGpuOgflops <= float(cpu[8])) and  (float(gpuLines[0].split(',')[8]) <= float(cpu[8])):
            gpuOnce.cpuGflops = 0.0
            gpuOnce.gpuGflops = 0.0
            gpuOnce.totalKib = 0.0
            gpuOnce.M = 0
            gpuOnce.N = 0
            gpuOnce.K = 0
    if(gpuAlways.M != 0 and float(cpu[8]) >= float(gpuA[8])):
        # Do check to see if this is a momentary drop that we should ignore
        if (prevGpuAgflops <= float(cpu[8])) and  (float(gpuLines[1].split(',')[8]) <= float(cpu[8])):
            gpuAlways.cpuGflops = 0.0
            gpuAlways.gpuGflops = 0.0
            gpuAlways.totalKib = 0.0
            gpuAlways.M = 0
            gpuAlways.N = 0
            gpuAlways.K = 0
    if("gemm" in kernel and gpuUnified.M != 0 and float(cpu[8]) >= float(gpuU[8])):
        # Do check to see if this is a momentary drop that we should ignore
        if (prevGpuUgflops <= float(cpu[8])) and  (float(gpuLines[2].split(',')[8]) <= float(cpu[8])):
            gpuUnified.cpuGflops = 0.0
            gpuUnified.gpuGflops = 0.0
            gpuUnified.totalKib = 0.0
            gpuUnified.M = 0
            gpuUnified.N = 0
            gpuUnified.K = 0
    # Update offload threshold if GPU.gflops > CPU.gflops
    if(gpuOnce.M == 0 and float(cpu[8]) < float(gpuO[8])):
        gpuOnce.cpuGflops = float(cpu[8])
        gpuOnce.gpuGflops = float(gpuO[8])
        gpuOnce.totalKib = float(gpuO[5])
        gpuOnce.M = int(gpuO[2])
        gpuOnce.N = int(gpuO[3])
        gpuOnce.K = int(gpuO[4])
    if(gpuAlways.M == 0 and float(cpu[8]) < float(gpuA[8])):
        gpuAlways.cpuGflops = float(cpu[8])
        gpuAlways.gpuGflops = float(gpuA[8])
        gpuAlways.totalKib = float(gpuA[5])
        gpuAlways.M = int(gpuA[2])
        gpuAlways.N = int(gpuA[3])
        gpuAlways.K = int(gpuA[4])
    if(gpuUnified.M == 0 and float(cpu[8]) < float(gpuU[8])):
        gpuUnified.cpuGflops = float(cpu[8])
        gpuUnified.gpuGflops = float(gpuU[8])
        gpuUnified.totalKib = float(gpuU[5])
        gpuUnified.M = int(gpuU[2])
        gpuUnified.N = int(gpuU[3])
        gpuUnified.K = int(gpuU[4])

    prevGpuAgflops = float(gpuA[8])
    prevGpuOgflops = float(gpuO[8])
    prevGpuUgflops = float(gpuU[8])

printResults(gpuOnce, gpuAlways, gpuUnified, strToKernel(kernel))