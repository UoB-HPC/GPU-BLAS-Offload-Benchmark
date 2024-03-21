import os
import matplotlib.pyplot as plt


# Check if CSV directory exists
path = os.path.join(os.getcwd(), 'CSV_Results')
if(not os.path.isdir(path)):
    print("ERROR - CSV_Results directory does not exist. Cannot generate any graphs.")
    exit(1)

# Get all filenames
path = os.path.join(os.getcwd(), 'CSV_Results')
filenames = os.listdir(path)

# Make Graphs directory
graphDir = os.path.join(os.getcwd(), 'Graphs')
if(not os.path.isdir(graphDir)):
    os.mkdir(graphDir)

# ------------------------------ GEMM Graphs --------------------------------------------
print("Creating GEMM graphs...")
# Create GEMM graphs
gemmFilenames = []
for i in range(0, len(filenames)):
    if "gemm_" in filenames[i]:
        gemmFilenames.append(filenames[i])

### CSV header format ==== Device,Kernel,M,N,K,Total Problem Size (KiB),Iterations,Total Seconds,GFLOP/s
for i in range(0, len(gemmFilenames)):
    mnk = []
    iters = 0
    kernel = ""
    cpu_Gflops = []
    gpuO_Gflops = []
    gpuA_Gflops = []
    gpuU_Gflops = []

    # Open file and get all lines
    fName = os.path.join(os.getcwd(), 'CSV_Results', gemmFilenames[i])
    openFile = open(fName, 'r')
    lines = openFile.readlines()
    lines.pop(0)
    if len(lines) == 0 :
        continue

    # Get number of iterations performed and kernel name
    line1 = lines[0].split(',')
    iters = int(line1[6])
    kernel = line1[1]

    # Get gflops (y-axis) and MNK values (x-axis) for CPU and all GPU types
    for line in lines:
        line = line.split(',')
        gflops = float(line[-1].rstrip())
        if line[0] == "cpu":
            cpu_Gflops.append(gflops)
            mnk.append([line[2], line[3], line[4]])
        elif line[0] == "gpu_offloadOnce":
            gpuO_Gflops.append(gflops)
        elif line[0] == "gpu_offloadAlways":
            gpuA_Gflops.append(gflops)
        elif line[0] == "gpu_unified":
            gpuU_Gflops.append(gflops)


    splitFileName = gemmFilenames[i].split('_')
    # Create x-axis label and tick values
    inputTypeStr = ""
    x_name = ""
    xVals = []
    if splitFileName[1] == "square.csv":
        x_name = "Value of M, N, K"
        inputTypeStr = "Square (M=N=K)"
        for j in range(0, len(mnk)):
            xVals.append(mnk[j][0])
    else :
        inputTypeStr = splitFileName[1][0].upper() + splitFileName[1][1:] + " " + splitFileName[2][:-4]
        probTypeStr = splitFileName[2][:-4]
        if probTypeStr == "16MxK":
            x_name = "Value of K, where M=16K and N=16K"
            for j in range(0, len(mnk)):
                xVals.append(mnk[j][2])
        elif probTypeStr == "32xK":
            x_name = "Value of K, where M=N=32"
            for j in range(0, len(mnk)):
                xVals.append(mnk[j][2])
        elif probTypeStr == "Mx16K":
            x_name = "Value of M and N, where K=16M"
            for j in range(0, len(mnk)):
                xVals.append(mnk[j][0])
        elif probTypeStr == "Mx32":
            x_name = "Value of M and N, where K=32"
            for j in range(0, len(mnk)):
                xVals.append(mnk[j][0])

    # Create y-axis label & graph title
    y_name = ""
    title = ""
    fp = ""
    if kernel == "sgemm" :
        fp = "FP32"
    elif kernel == "dgemm":
        fp = "FP64"
    y_name = "{} GFLOP/s".format(fp)        
    title = "{}GEMM Performance for {} Style Inputs - {} iterations per problem size".format(kernel[0].upper(), inputTypeStr, iters)

    # Make Graph
    fig1 = plt.figure(figsize=(25,14))
    ax1 = fig1.add_subplot()

    ax1.plot(xVals, cpu_Gflops, color="#332288", marker=".", label="CPU")
    ax1.plot(xVals, gpuO_Gflops, color="#44AA99", marker="x", label="GPU (Offload Once)")
    ax1.plot(xVals, gpuA_Gflops, color="#CC6677", marker="+", label="GPU (Offload Always)")
    ax1.plot(xVals, gpuU_Gflops, color="#DDCC77", marker=">", label="GPU (Unified Memory)")

    # Set X ticks
    NUM_TICK = 8
    numXVals = len(xVals)
    if numXVals < NUM_TICK:
        # Print all labels
        plt.xticks(ticks=range(0, numXVals, 1), labels=xVals)
    else:
        # Calculate labels
        locInterval = int((numXVals) / (NUM_TICK-1))
        tickLocs = [0]
        for q in range(1, (NUM_TICK-1)):
            tickLocs.append(1 + (locInterval * q))
        tickLocs.append(numXVals - 1)

        labelInterval = int((int(xVals[-1]) - int(xVals[0])) / (NUM_TICK-1))
        tickLabs = [xVals[0]]
        for q in range(1, (NUM_TICK-1)):
            tickLabs.append(int(xVals[0]) + (labelInterval * q))
        tickLabs.append(int(xVals[-1]))

        plt.xticks(ticks=tickLocs, labels=tickLabs)

    # Force setting of y-axis labels. If this isn't done then the range is weird...
    yLoc, yLab = plt.yticks()
    yLoc = yLoc.tolist()
    # Remove negative first element of the list
    if yLoc[0] != 0:
        yLoc = yLoc[1:]
    plt.ylim(0, yLoc[-1])
    plt.yticks(ticks=yLoc)

    plt.margins(x=0.01, y=0.01)
    plt.legend(loc='upper left', fancybox=True, ncol = 1)
    plt.xlabel(x_name, fontsize=16)
    plt.ylabel(y_name, fontsize=16)
    plt.title(title, fontsize=20)
    plt.savefig(fname="{}/{}.png".format(graphDir, gemmFilenames[i][:-4]), format="png", dpi=100)
    

print("Finished!")
# ---------------------------------------------------------------------------------------