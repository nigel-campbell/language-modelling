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
parser.add_argument('--emsize', type=str, default=10, help='embedding size')
args = parser.parse_args()

path = args.data
verbose = args.verbose
embed_size = args.emsize

corpus = data.Corpus(path, verbose)

vocab_size = len(corpus)

if verbose:
    print('Read {} tokens'.format(vocab_size))
model = model.Model(vocab_size, embed_size)

print(corpus.data)
print(model(corpus.data[0]))
