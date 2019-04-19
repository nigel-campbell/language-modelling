import torch
import argparse
import model
import data

description = """
Custom driver for evaluating various neural
architectures for NLP language models.
"""
parser = argparse.ArgumentParser(description=description)
parser.add_argument('--data', type=str, required=True, help='location of the data corpus')
parser.add_argument('--verbose', action='store_true', help='Enables verbose output.')
parser.add_argument('--emsize', type=int, default=200, help='embedding size')
parser.add_argument('--nhidden', type=int, default=10, help='number hidden')
parser.add_argument('--nlayers', type=int, default=2, help='number of layers' )
parser.add_argument('--epochs', type=int, default=100, help='number of epochs')
args = parser.parse_args()

path = args.data
verbose = args.verbose
embed_size = args.emsize
nhidden = args.nhidden
nlayers = args.nlayers
epochs = args.epochs

corpus = data.Corpus(path, verbose)

vocab_size = len(corpus)

if verbose:
    print('Read {} tokens'.format(vocab_size))
lm = model.Model(vocab_size, embed_size, nhidden, nlayers)
losses = lm.fit(corpus, epochs)
print(losses)

