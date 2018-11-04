import argparse, os, sys
import numpy as np
from scipy import stats
import pandas as pd

# If you write your classes in other Python files
# you can import them here.

def readFile(filename):
    with open(filename, 'r') as f:
        data = pd.read_table(f)
        data = data.drop(data.columns[0],axis=1) #remove time col header
        data = data.values #keep arr only
        num_rows = data.shape[0] # conditions
        num_cols = data.shape[1] # gene
    return (data, num_rows, num_cols)

def printSorted(input_MI, output_file_path):
    MI_s1 = sorted(input_MI.items(), key=lambda x: (x[0][0], x[0][1])) #sort by bin ind
    sorted_MI = sorted(MI_s1, key=lambda x: x[1],reverse=True) #sort desc by val
    with open(output_file_path,'w') as fout:
        for b,v in sorted_MI:
            print("({},{})\t{:.3f}".format(b[0]+1,b[1]+1,v), file=fout)

def uniDensiProcess(data_bins, bin_num, num_genes, num_condi, output_file_path):
    main_MI = {}
    sub_MI = np.zeros((num_genes, num_genes))
    count_pr = np.zeros((bin_num, num_genes))
    data_bins = data_bins.astype(int) #ind

    for i in range(num_condi):
        for j in range(num_genes):
            count_pr[data_bins[i,j],j] += 1
    count_pr += 0.1 #add pseudoct
    
    for i in range(num_genes):
        for j in range(i+1, num_genes):
            combi_ct = np.zeros((bin_num, bin_num))
            for k in range(num_condi):
                combi_ct[data_bins[k,i], data_bins[k,j]] += 1
            combi_ct += 0.1
            prob_ij = combi_ct / np.sum(combi_ct)
            prob_i = count_pr[:, i] / np.sum(count_pr[:, i])
            prob_j = count_pr[:, j] / np.sum(count_pr[:, j])
            
            for m in range(bin_num):
                for n in range(bin_num):
                    sub_MI[i,j] += prob_ij[m,n] * np.log2(prob_ij[m,n] / (prob_i[m] * prob_j[n]))

            main_MI[(i,j)] = np.round(sub_MI[i,j], decimals=3)
    printSorted(main_MI, output_file_path)

def main(args):
    # Parse input arguments
    data_file_path = args.dataset
    bin_num = args.bin_num
    output_file_path = args.out
    bin_str = args.str.lower()

    # Where you run your code.
    data, num_condi, num_genes = readFile(data_file_path)
    data_bins = np.zeros((num_condi, num_genes))

    if bin_str == 'uniform':

        for i in range(num_genes):
            min_now = min( data[:,i] )
            max_now = max( data[:,i] )
            stepsz = (max_now - min_now) / bin_num
            #loop bin
            dataslot = []
            for j in range(1, bin_num): #[ )
                dataslot.append(stepsz * j + min_now )
            data_bins[:,i] = np.digitize(data[:,i], dataslot, right = True)
        uniDensiProcess(data_bins, bin_num, num_genes, num_condi, output_file_path)

    elif bin_str == 'density':

        for i in range(num_genes):
            stepsz = 100 / bin_num
            dataslot=[]
            for j in range(1, bin_num):
                dataslot.append( np.percentile(data[:,i], j * stepsz) )
            data_bins[:,i] = np.digitize(data[:,i], dataslot, right=True)
        uniDensiProcess(data_bins, bin_num, num_genes, num_condi, output_file_path)

    elif bin_str == 'kernel':
        MI_mat = {}
        xyconst = 0.012*0.012
        for i in range(num_genes):
            for j in range(i+1,num_genes):
                point = 0
                #docs
                Xgrid,Ygrid = np.mgrid[-0.1:1.1:100j, -0.1:1.1:100j]
                positions = np.vstack([Xgrid.ravel(), Ygrid.ravel()])
                values = np.vstack([data[:,i], data[:,j]])
                kernel_ij = stats.gaussian_kde(values)
                Z = np.reshape(kernel_ij(positions).T, Xgrid.shape)
                Z += 0.001
                for k in range(100):
                    for l in range(100):
                        prob_ij = Z[k,l] * xyconst
                        prob_i = np.sum(Z[k,:]) * xyconst
                        prob_j = np.sum(Z[:,l]) * xyconst
                        point += prob_ij * np.log2(prob_ij / (prob_i * prob_j))

                MI_mat[(i,j)] = np.round(point, decimals=3)

        printSorted(MI_mat, output_file_path)

# Note: this syntax checks if the Python file is being run as the main program
# and will not execute if the module is imported into a different module
if __name__ == "__main__":
   # Note: this example shows named command line arguments.  See the argparse
   # documentation for positional arguments and other examples.
   parser = argparse.ArgumentParser(
       description=__doc__,
       formatter_class=argparse.RawDescriptionHelpFormatter)
   parser.add_argument('--dataset',
                       help='input gene expression data file path',
                       type=str,
                       default='')
   parser.add_argument('--bin_num',
                       help='number of bins, not used with the kernel binning strategy',
                       type=int,
                       default=2)
   parser.add_argument('--out',
                       help='MI output file path',
                       type=str,
                       default='')
   parser.add_argument('--str',
                       help='binning strategy',
                       type=str,
                       choices={'uniform', 'density', 'kernel'},
                       default='uniform')

   args = parser.parse_args()
   main(args)
