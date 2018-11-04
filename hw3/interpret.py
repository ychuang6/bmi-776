# Adapted by: Anthony Gitter
# Adapted from https://github.com/kundajelab/dragonn/blob/master/dragonn/__main__.py

from dragonn.utils import encode_fasta_sequences
from dragonn.models import SequenceDNN
from dragonn.plot import plot_motif, add_letters_to_axis
import matplotlib.pyplot as plt
import argparse

# Adapted from https://github.com/kundajelab/dragonn/blob/master/dragonn/tutorial_utils.py
# Identical to GitHub version except it saves figure to disk
def plot_sequence_filters(model, prefix):
    '''
    model - a trained SequenceDNN
    prefix - the name of the loaded model
    '''
    fig = plt.figure(figsize=(15, 8))
    fig.subplots_adjust(hspace=0.1, wspace=0.1)
    conv_filters = model.get_sequence_filters()
    num_plots_per_axis = int(len(conv_filters)**0.5) + 1
    for i, conv_filter in enumerate(conv_filters):
        ax = fig.add_subplot(num_plots_per_axis, num_plots_per_axis, i+1)
        add_letters_to_axis(ax, conv_filter.T)
        ax.axis('off')
        ax.set_title('Filter %s' % (str(i+1)))
    fig.savefig('{}_convolutional_filters.png'.format(prefix), bbox_inches='tight')


def main(args):
    '''
    args - parsed arguments that include pos_sequences, neg_sequences,
    arch_file, and weights_file
    '''
    # encode fasta
    print('Loading sequence data...')
    pos_seq = encode_fasta_sequences(args.pos_sequences)
    print('{} positive test sequences'.format(len(pos_seq)))
    neg_seq = encode_fasta_sequences(args.neg_sequences)
    print('{} negative test sequences\n'.format(len(neg_seq)))

    # load model
    prefix = args.arch_file.replace('.arch.json', '')
    print('Loading {} model...'.format(prefix))
    model = SequenceDNN.load(args.arch_file, args.weights_file)
    
    # predict binding probability on test sequences
    print('Getting predictions...')
    pos_predictions = model.predict(pos_seq)
    for index, pred in enumerate(pos_predictions):
        print('positive_test_{}\tP(bound)={}'.format(index, pred[0]))
    print('')
    neg_predictions = model.predict(neg_seq)
    for index, pred in enumerate(neg_predictions):
        print('negative_test_{}\tP(bound)={}'.format(index, pred[0]))
    print('')

    # visualize trained model and motifs
    print('Plotting deeplift scores on positive sequences...')
    model.plot_deeplift(pos_seq, '{}_deeplift_positive'.format(prefix))

    print('Plotting true motifs...')
    motif_names = ['IRF_known1','NFKB_known1']
    for index, motif in enumerate(motif_names):
        fig = plot_motif(motif, figsize=(10, 4), ylab=motif)
        fig.savefig('motif{}.png'.format(index + 1), bbox_inches='tight')

    print('Plotting architecture...')
    model.plot_architecture('{}_architecture.png'.format(prefix))

    print('Plotting convolutional filters...')
    plot_sequence_filters(model, prefix)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--pos-sequences', type=str, required=True,
                                   help='fasta with positive sequences')
    parser.add_argument('--neg-sequences', type=str, required=True,
                                   help='fasta with negative sequences')
    parser.add_argument('--arch-file', type=str, required=True,
                                    help='model json file')
    parser.add_argument('--weights-file', type=str, required=True,
                                    help='weights hd5 file')

    args = parser.parse_args()

    main(args)
