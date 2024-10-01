import os
import sys
import matplotlib.pyplot as plt



directory = "CSV_Results"
# Get given CSV file directory
if(len(sys.argv) > 1):
    directory = sys.argv[1]

outputDir = "Graphs_" + directory.replace('/', '_')

# Check if CSV directory exists
path = os.path.join(os.getcwd(), directory)
if(not os.path.isdir(path)):
    print("ERROR - {} directory does not exist. Cannot generate any graphs.".format)
    exit(1)

# Get all filenames
path = os.path.join(os.getcwd(), directory)
filenames = os.listdir(path)

# Make Graphs directory
graphDir = os.path.join(os.getcwd(), outputDir)
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
    fName = os.path.join(os.getcwd(), directory, gemmFilenames[i])
    openFile = open(fName, 'r')
    lines = openFile.readlines()
    lines.pop(0) # Remove headers
    if len(lines) == 0 :
        continue

    # Get number of iterations performed and kernel name
    line1 = lines[0].split(',')
    iters = int(line1[6])
    kernel = line1[1]

    # Get gflops (y-axis) and MNK values (x-axis) for CPU and all GPU types
    for line in lines:
        line = line.split(',')
        # Get MNK
        if (len(mnk) == 0) or ([line[2], line[3], line[4]] not in mnk):
            mnk.append([line[2], line[3], line[4]])
        # Get Gflops
        gflops = float(line[-1].rstrip())
        if line[0] == "cpu":
            cpu_Gflops.append(gflops)
        elif line[0] == "gpu_offloadOnce":
            gpuO_Gflops.append(gflops)
        elif line[0] == "gpu_offloadAlways":
            gpuA_Gflops.append(gflops)
        elif line[0] == "gpu_unified":
            gpuU_Gflops.append(gflops)

    # Create x-axis label and tick values
    inputTypeStr = ""
    x_name = ""
    xVals = []
    if "_square_square_M=N=K" in gemmFilenames[i]:
        x_name = "Value of M, N, K"
        inputTypeStr = "Square x Square (M=N=K)"
        for j in range(0, len(mnk)):
            xVals.append(mnk[j][0])
    elif "_tall-thin_short-wide_M=N_M=16K" in gemmFilenames[i]:
        x_name = "Value of K where M=16K and N=16K"
        inputTypeStr = "Tall-Thin x Short-Wide (M=N=16K)"
        for j in range(0, len(mnk)):
            xVals.append(mnk[j][2])
    elif "_tall-thin_short-wide_M=N_K=32" in gemmFilenames[i]:
        x_name = "Value of M and N, where K=32"
        inputTypeStr = "Tall-Thin x Short-Wide (M=N, K=32)"
        for j in range(0, len(mnk)):
            xVals.append(mnk[j][0])
    elif "_short-wide_tall-thin_M=N_K=16M" in gemmFilenames[i]:
        x_name = "Value of M and N, where K=16M"
        inputTypeStr = "Short-Wide x Tall-Thin (M=N, K=16M)"
        for j in range(0, len(mnk)):
            xVals.append(mnk[j][0])
    elif "_short-wide_tall-thin_M=N=32_K" in gemmFilenames[i]:
        x_name = "Value of K, where M=32 and N=32"
        inputTypeStr = "Short-Wide x Tall-Thin (M=N=32, K)"
        for j in range(0, len(mnk)):
            xVals.append(mnk[j][2])
    elif "_tall-thin_square_K=N_M=16K" in gemmFilenames[i]:
        x_name = "Value of N and K, where M=16K"
        inputTypeStr = "Tall-Thin x Square (N=K, M=16K)"
        for j in range(0, len(mnk)):
            xVals.append(mnk[j][2])
    elif "_tall-thin_square_K=N=32_M" in gemmFilenames[i]:
        x_name = "Value of M, where N=32 and K=32"
        inputTypeStr = "Tall-Thin x Square (M, N=K=32)"
        for j in range(0, len(mnk)):
            xVals.append(mnk[j][0])
    elif "_square_short-wide_M=K_N=16K" in gemmFilenames[i]:
        x_name = "Value of M and K, where N=16K"
        inputTypeStr = "Square x Short-Wide (M=K, N=16K)"
        for j in range(0, len(mnk)):
            xVals.append(mnk[j][0])
    elif "_square_short-wide_M=K=32_N" in gemmFilenames[i]:
        x_name = "Value of N, where M=32 and K=32"
        inputTypeStr = "Square x Short-Wide (M=K=32, N)"
        for j in range(0, len(mnk)):
            xVals.append(mnk[j][1])
    elif "_sparse_square" in gemmFilenames[i]:
        x_name = "Value of M, N, K"
        inputTypeStr = "Sparse square matrices"
        for j in range(0, len(mnk)):
            xVals.append(mnk[j][0])
    else:
        # File not supported so go to next file
        continue



    # Create y-axis label & graph title
    y_name = ""
    title = ""
    fp = ""
    if kernel == "sgemm" :
        fp = "FP32"
    elif kernel == "dgemm":
        fp = "FP64"
    y_name = "{} GFLOP/s".format(fp)        
    title = "{}GEMM Performance for {} Problems - {} iterations per problem size".format(kernel[0].upper(), inputTypeStr, iters)

    # Make Graph
    fig1 = plt.figure(figsize=(28,16))
    ax1 = fig1.add_subplot()

    gpuEnabled = False
    if len(cpu_Gflops) > 0:
        ax1.plot(xVals, cpu_Gflops, color="#332288", marker=".", label="CPU")
        # Plot line at max GFLOP/s
        yCoord = round(max(cpu_Gflops),1)
        ax1.axhline(yCoord, color='black', linestyle='--')
        ax1.text(x=0, y=yCoord, s="Max CPU GFLOP/s : {:,}".format(yCoord), fontsize=12, ha='left', va='bottom')
    if len(gpuO_Gflops) > 0:
        ax1.plot(xVals, gpuO_Gflops, color="#44AA99", marker="x", label="GPU (Offload Once)")
        gpuEnabled = True
    if len(gpuA_Gflops) > 0:
        ax1.plot(xVals, gpuA_Gflops, color="#CC6677", marker="+", label="GPU (Offload Always)")
        gpuEnabled = True
    if len(gpuU_Gflops) > 0:
        ax1.plot(xVals, gpuU_Gflops, color="#DDCC77", marker=">", label="GPU (Unified Memory)")
        gpuEnabled = True

    if(gpuEnabled):
        yCoord = round(max([max(gpuO_Gflops), max(gpuA_Gflops), max(gpuU_Gflops)]) ,1)
        ax1.axhline(yCoord, color='black', linestyle='--')
        ax1.text(x=0, y=yCoord, s="Max GPU GFLOP/s : {:,}".format(yCoord), fontsize=12, ha='left', va='bottom')

    # Set X ticks
    NUM_TICK = 8
    numXVals = len(xVals)
    if numXVals < NUM_TICK:
        # Print all labels
        plt.xticks(ticks=range(0, numXVals, 1), labels=xVals, fontsize=20)
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

        plt.xticks(ticks=tickLocs, labels=tickLabs, fontsize=20)

    # Force setting of y-axis labels. If this isn't done then the range is weird...
    yLoc, yLab = plt.yticks()
    yLoc = yLoc.tolist()
    # Remove negative first element of the list
    if yLoc[0] != 0:
        yLoc = yLoc[1:]
    plt.ylim(0, yLoc[-1])
    plt.yticks(ticks=yLoc, fontsize=20)

    plt.margins(x=0.01, y=0.01)
    leg = plt.legend(loc='upper left', fancybox=True, ncol = 2, fontsize=18)
    for obj in leg.legend_handles:
        obj.set_linewidth(3.0)
        obj.set_markersize(15.0)
        obj.set_markeredgewidth(3.0)
    plt.xlabel(x_name, fontsize=20)
    plt.ylabel(y_name, fontsize=20)
    plt.title(title, fontsize=20)
    plt.savefig(fname="{}/{}.png".format(graphDir, gemmFilenames[i][:-4]), format="png", dpi=100, bbox_inches="tight")
    plt.close('all')
    

print("Finished!")
# ---------------------------------------------------------------------------------------

# ------------------------------ GEMV Graphs --------------------------------------------
print("Creating GEMV graphs...")
# Create GEMV graphs
gemvFilenames = []
for i in range(0, len(filenames)):
    if "gemv_" in filenames[i]:
        gemvFilenames.append(filenames[i])

### CSV header format ==== Device,Kernel,M,N,K,Total Problem Size (KiB),Iterations,Total Seconds,GFLOP/s
for i in range(0, len(gemvFilenames)):
    mn = []
    iters = 0
    kernel = ""
    cpu_Gflops = []
    gpuO_Gflops = []
    gpuA_Gflops = []
    gpuU_Gflops = []

    # Open file and get all lines
    fName = os.path.join(os.getcwd(), directory, gemvFilenames[i])
    openFile = open(fName, 'r')
    lines = openFile.readlines()
    lines.pop(0) # Remove headers
    if len(lines) == 0 :
        continue

    # Get number of iterations performed and kernel name
    line1 = lines[0].split(',')
    iters = int(line1[6])
    kernel = line1[1]

    # Get gflops (y-axis) and MN values (x-axis) for CPU and all GPU types
    for line in lines:
        line = line.split(',')
        # Get MN
        if (len(mn) == 0) or ([line[2], line[3]] not in mn):
            mn.append([line[2], line[3]])
        # Get Gflops
        gflops = float(line[-1].rstrip())
        if line[0] == "cpu":
            cpu_Gflops.append(gflops)
        elif line[0] == "gpu_offloadOnce":
            gpuO_Gflops.append(gflops)
        elif line[0] == "gpu_offloadAlways":
            gpuA_Gflops.append(gflops)
        elif line[0] == "gpu_unified":
            gpuU_Gflops.append(gflops)


    # Create x-axis label and tick values
    inputTypeStr = ""
    x_name = ""
    xVals = []
    if "_square_vector_M=N" in gemvFilenames[i]:
        x_name = "Value of M, N"
        inputTypeStr = "Square x Vector (M=N)"
        for j in range(0, len(mn)):
            xVals.append(mn[j][0])
    elif "_tall-thin_vector_M=16N" in gemvFilenames[i]:
        x_name = "Value of N where M=16N"
        inputTypeStr = "Tall-Thin x Vector (M=16N)"
        for j in range(0, len(mn)):
            xVals.append(mn[j][1])
    elif "_tall-thin_vector_M_N=32" in gemvFilenames[i]:
        x_name = "Value of M, where N=32"
        inputTypeStr = "Tall-Thin x Vector (M, N=32)"
        for j in range(0, len(mn)):
            xVals.append(mn[j][0])
    elif "_short-wide_vector_N=16M" in gemvFilenames[i]:
        x_name = "Value of M, where N=16M"
        inputTypeStr = "Short-Wide x Vector (N=16M)"
        for j in range(0, len(mn)):
            xVals.append(mn[j][0])
    elif "_short-wide_vector_M=32_N" in gemvFilenames[i]:
        x_name = "Value of N, where M=32"
        inputTypeStr = "Short-Wide x Vector (M=32, N)"
        for j in range(0, len(mn)):
            xVals.append(mn[j][1])
    else:
        # File not supported so go to next file
        continue



    # Create y-axis label & graph title
    y_name = ""
    title = ""
    fp = ""
    if kernel == "sgemv" :
        fp = "FP32"
    elif kernel == "dgemv":
        fp = "FP64"
    y_name = "{} GFLOP/s".format(fp)        
    title = "{}GEMV Performance for {} Problems - {} iterations per problem size".format(kernel[0].upper(), inputTypeStr, iters)

    # Make Graph
    fig1 = plt.figure(figsize=(28,16))
    ax1 = fig1.add_subplot()

    gpuEnabled = False
    if len(cpu_Gflops) > 0:
        ax1.plot(xVals, cpu_Gflops, color="#332288", marker=".", label="CPU")
        # Plot line at max GFLOP/s
        yCoord = round(max(cpu_Gflops),1)
        ax1.axhline(yCoord, color='black', linestyle='--')
        ax1.text(x=0, y=yCoord, s="Max CPU GFLOP/s : {:,}".format(yCoord), fontsize=12, ha='left', va='bottom')
    if len(gpuO_Gflops) > 0:
        ax1.plot(xVals, gpuO_Gflops, color="#44AA99", marker="x", label="GPU (Offload Once)")
        gpuEnabled = True
    if len(gpuA_Gflops) > 0:
        ax1.plot(xVals, gpuA_Gflops, color="#CC6677", marker="+", label="GPU (Offload Always)")
        gpuEnabled = True
    if len(gpuU_Gflops) > 0:
        ax1.plot(xVals, gpuU_Gflops, color="#DDCC77", marker=">", label="GPU (Unified Memory)")
        gpuEnabled = True

    if(gpuEnabled):
        yCoord = round(max([max(gpuO_Gflops), max(gpuA_Gflops), max(gpuU_Gflops)]) ,1)
        ax1.axhline(yCoord, color='black', linestyle='--')
        ax1.text(x=0, y=yCoord, s="Max GPU GFLOP/s : {:,}".format(yCoord), fontsize=12, ha='left', va='bottom')

    # Set X ticks
    NUM_TICK = 8
    numXVals = len(xVals)
    if numXVals < NUM_TICK:
        # Print all labels
        plt.xticks(ticks=range(0, numXVals, 1), labels=xVals, fontsize=20)
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

        plt.xticks(ticks=tickLocs, labels=tickLabs, fontsize=20)

    # Force setting of y-axis labels. If this isn't done then the range is weird...
    yLoc, yLab = plt.yticks()
    yLoc = yLoc.tolist()
    # Remove negative first element of the list
    if yLoc[0] != 0:
        yLoc = yLoc[1:]
    plt.ylim(0, yLoc[-1])
    plt.yticks(ticks=yLoc, fontsize=20)

    plt.margins(x=0.01, y=0.01)
    leg = plt.legend(loc='upper left', fancybox=True, ncol = 2, fontsize=18)
    for obj in leg.legendHandles:
        obj.set_linewidth(3.0)
        obj.set_markersize(15.0)
        obj.set_markeredgewidth(3.0)
    plt.xlabel(x_name, fontsize=20)
    plt.ylabel(y_name, fontsize=20)
    plt.title(title, fontsize=20)
    plt.savefig(fname="{}/{}.png".format(graphDir, gemvFilenames[i][:-4]), format="png", dpi=100, bbox_inches="tight")
    plt.close('all')
    

print("Finished!")
# ---------------------------------------------------------------------------------------