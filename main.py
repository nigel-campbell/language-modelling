import torch
import argparse
import model
import data
import csv
from metrics import Metrics

description = """
Custom driver for evaluating various neural
architectures for generative language models.
"""
parser = argparse.ArgumentParser(description=description)
parser.add_argument('--data', type=str, required=True, help='location of the data corpus')
parser.add_argument('--arch', type=str, default='LSTM', help='Model arch (RNN,LSTM,GRU)')
parser.add_argument('--emsize', type=int, default=200, help='embedding size')
parser.add_argument('--nhidden', type=int, default=10, help='number hidden')
parser.add_argument('--nlayers', type=int, default=2, help='number of layers' )
parser.add_argument('--epochs', type=int, default=100, help='number of epochs')
parser.add_argument('--metrics', type=str, default='metrics/', help='directory for storing metrics.')
parser.add_argument('--mdout', type=str, default='model.pt', help='file output of serialized model')
parser.add_argument('--verbose', action='store_true', help='Enables verbose output.')
parser.add_argument('--cuda', action='store_true', help='Enables cuda')
args = parser.parse_args()

path = args.data
verbose = args.verbose
embed_size = args.emsize
nhidden = args.nhidden
nlayers = args.nlayers
epochs = args.epochs
metrics_dir = args.metrics
model_out = args.mdout
arch = args.arch
cuda = args.cuda

corpus = data.Corpus(path, verbose)

vocab_size = len(corpus)

device = torch.device("cpu" if not cuda else "cuda")

lm = model.Model(vocab_size, embed_size, nhidden, nlayers, model=model, cuda=cuda)
lm = lm.to(device)
losses = lm.fit(corpus, epochs)
lm.metrics.save()

with open(mdout,'wb') as f:
    torch.save(lm, f)

sentence = lm.generate("Make", corpus)
