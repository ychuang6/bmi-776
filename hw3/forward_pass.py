# Assume that a network has been trained using DragoNN and the network weights are available. 
# The network includes 2 convolutional layers, a max-pooling layer, and a fully connected layer in sequence. 
# Each convolutional layer contains 5 filters, has window size and stride size equal to 15 and 1, respectively, 
# and uses ReLU as its activation function. The max-pooling layer has window size and stride size both equal to 35. 
# The fully connected layer uses the sigmoid function as its activation function.
# Write a program forward_pass.py that reads in positive_test.fa, negative_test.fa, and weights files 
# and outputs the probability of each test sequence being bound. Transform the sequences using one-hot encoding. 

import argparse, os, sys, math
import pandas as pd
import numpy as np

def readFile(filename):
    seqs = []
    seqBinarized = []
    headers =[]
    with open(filename,'r') as f:
        for line in f.readlines():
            if line.startswith('>'):
                headers.extend(line.strip())
            else:
                seq = line.strip()
                seqs.append(seq)
                seqBinarized.append(createBin(seq))
                
    return headers,seqs,seqBinarized

def readWeights(filename):
    weightdata = []
    with open(filename,'r') as f:
        for line in f.readlines():
            wline = list( map(float, line.strip().split('\t')))
            weightdata.append(wline)
        weightdata = np.asmatrix(weightdata)
    return weightdata

def createBin(seq):
    # create binary matrix of 4 x seq_length of the input seq
    l = np.array(list(seq))
    d = np.array(['A','C','G','T'])
    indexMat = d[:,np.newaxis] == l
    indexMat = indexMat.astype(int)
    indexMat = np.asmatrix(indexMat)
    
    return indexMat

def conv_single(seq_In, conv_w, conv_b):
    # single convolution function 
    windowsize = 15
    (height4, L) = seq_In.shape   #height has to be 4
    #f = len(conv_w) # has to be 5
    
    conW = L - windowsize + 1
    
    # Initialize the output Z 
    Z = np.asmatrix( np.zeros((5, conW)) )
    
    for i in range(conW):
        for j in range(5): #5
            s = np.multiply(seq_In[:,i:i+windowsize], conv_w[j])
            Zs = np.sum(s) + conv_b[j]
            Z[j,i] = max(0, Zs)

    return Z

def forwardpass(Z2, dense_w, dense_b):
    (height5, L) = Z2.shape
    
    tempZ = np.asmatrix(np.zeros( [height5, int(L/35)]))

    (tH, tL) = tempZ.shape
    for i in range(tL):
        if i == tL-1:
            tempZ[:,i] = np.amax(Z2[:,i*35:], axis=1)
        else:
            tempZ[:,i] = np.amax(Z2[:,i*35: (i*35)+35], axis=1)

    ZZ = np.multiply( tempZ.reshape(tH*tL,1), dense_w)
    Z = np.sum(ZZ) + dense_b
    probZ = 1 / (1 + math.exp(-Z))
    
    return probZ


def main(args):
    # Parse input arguments
    posfile = args.positive_sequences
    negfile = args.negative_sequences
    weight_Dir = args.weights
    output_file = args.out

    headposi, posi, posi_mat = readFile(posfile)
    headnega, nega, nega_mat = readFile(negfile)

    weightfile_b1 = os.path.join(weight_Dir,'conv1_bias.txt')
    weightfile_b2 = os.path.join(weight_Dir,'conv2_bias.txt')
    conv_b1 = readWeights(weightfile_b1)
    conv_b2 = readWeights(weightfile_b2)

    conv_w1=[]
    conv_w2=[]

    for i in range(1,6):
        weightfile1 = os.path.join(weight_Dir,'conv1_ch{}.txt'.format(i))
        weightfile2 = os.path.join(weight_Dir,'conv2_ch{}.txt'.format(i))
        conv_now1 = readWeights(weightfile1)
        conv_now2 = readWeights(weightfile2)
        conv_w1.append(conv_now1)
        conv_w2.append(conv_now2)

    dense_bfile=os.path.join(weight_Dir,'dense_bias.txt')
    dense_wfile=os.path.join(weight_Dir,'dense_wgts.txt')
    dense_b = readWeights(dense_bfile)
    dense_w = readWeights(dense_wfile)

    with open(output_file, 'w') as f:
        f.write("Positive Sequences:\n")
        for seq_i in posi_mat:
            Z1 = conv_single(seq_i, conv_w1, conv_b1)
            Z2 = conv_single(Z1, conv_w2, conv_b2)
            Zprob = forwardpass(Z2, dense_w, dense_b)
            print('{}'.format(Zprob),file=f)
            
        f.write("\nNegative Sequences:\n")
        for seq_i in nega_mat:
            Z1 = conv_single(seq_i, conv_w1, conv_b1)
            Z2 = conv_single(Z1, conv_w2, conv_b2)
            Zprob = forwardpass(Z2, dense_w, dense_b)
            print('{}'.format(Zprob),file=f)

if __name__ == "__main__":
    # Note: this example shows named command line arguments.  See the argparse
    # documentation for positional arguments and other examples.
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--positive_sequences',
                        help='positive sequence fasta file path',
                        type=str,
                        default='')
    parser.add_argument('--negative_sequences',
                        help='negative sequences fasta file path',
                        type=str,
                        default='')
    parser.add_argument('--weights',
                        help='weight files directory',
                        type=str,
                        default='')
    parser.add_argument('--out',
                        help='output file name',
                        type=str,
                        default='')

    args = parser.parse_args()

    main(args)
