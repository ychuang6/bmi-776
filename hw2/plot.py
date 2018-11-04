import matplotlib
matplotlib.use('Agg')
import argparse, os, sys
import matplotlib.pyplot as plt
import numpy as np


def main(args):
    # Parse input arguments
    MI_file_path = args.MI
    gold_file_path = args.gold
    image_file_path = args.name + ".png"

    predicted_pairs = []        # predicted edges sorted by MI in decreasing order
    true_pairs = []             # true edges

    with open(MI_file_path, 'r') as f:
        lines = f.readlines()

    for line in lines:
        line = line.split()
        pair = line[0].split(',')
        predicted_pairs.append([pair[0][1:], pair[1][:-1]])
    
    with open(gold_file_path, 'r') as f:
        standard = f.readlines()
    
    for line in lines:
        line = line.split()
        true_pairs.append([line[0], line[1]])

    FPR = []
    TPR = []

    # TODO: Calculate the coordinates of points on ROC
    # store the x coordinates in FPR
    # store the y coordinates in TPR

    storeTF = []

    for pa in predicted_pairs:
        if pa in true_pairs: #if pairnow is in goldstd then add 1
            storeTF.append(1)
        else: 
            storeTF.append(-1)

    countT = storeTF.count(1)
    countF = storeTF.count(-1)

    TPR.append(0)
    FPR.append(0)
    t1 = 0
    f1 = 0
    for sl in storeTF: #sort pos&neg into TPR/FPR bins
        if sl==1:  
            t1 += 1
        else:
            f1 += 1
        TPR.append(t1 / countT)
        FPR.append(f1 / countF)

    TPR.append(1)
    FPR.append(1)

    
    # Compute AUROC with the trapezoidal rule
    area = np.trapz(y=TPR, x=FPR)
    
    fig = plt.figure()
    plt.plot(FPR, TPR, '-')
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.title('ROC Curve (AUROC = {0:.3f})'.format(area))
    plt.axis([0, 1, 0, 1])
    
    # Increase the image resolution to 300 dots per inch
    fig.savefig(image_file_path, dpi=300)

# Note: this syntax checks if the Python file is being run as the main program
# and will not execute if the module is imported into a different module
if __name__ == "__main__":
    # Note: this example shows named command line arguments.  See the argparse
    # documentation for positional arguments and other examples.
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--MI',
                        help='MI file path.',
                        type=str,
                        default='')
    parser.add_argument('--gold',
                        help='gold standard network file path.',
                        type=str,
                        default='')
    parser.add_argument('--name',
                        help='output image file name without the file type.',
                        type=str,
                        default='')

    args = parser.parse_args()

    main(args)

# filefolder='/Users/seagull/Box Sync/Class Spring 2018/BMI776/Homework/HW2/hw2_files'
# args = argparse.ArgumentParser()
# args.MI = os.path.join(filefolder, 'toy_data_kern_out.txt')
# args.gold = os.path.join(filefolder, 'toy_network.txt')
# args.name = os.path.join(filefolder, 'roc_kernel_toy')
# main(args)
