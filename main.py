import torch
import argparse
import model
import data
import csv
import os
from metrics import Metrics

description = """
Custom driver for evaluating various neural
architectures for generative language models.
"""
parser = argparse.ArgumentParser(description=description)
parser.add_argument('--data', type=str, default='data/2013.txt', help='location of the data corpus')
parser.add_argument('--arch', type=str, default='LSTM', help='Model arch (RNN,LSTM,GRU)')
parser.add_argument('--emsize', type=int, default=200, help='embedding size')
parser.add_argument('--nhidden', type=int, default=10, help='number hidden')
parser.add_argument('--nlayers', type=int, default=2, help='number of layers' )
parser.add_argument('--epochs', type=int, default=15, help='number of epochs')
parser.add_argument('--lookahead', type=int, default=2, help='number of lookahead steps')
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
lookahead = args.lookahead

corpus = data.Corpus(path, verbose)

vocab_size = len(corpus)

device = torch.device("cpu" if not cuda else "cuda")

lm = model.Model(vocab_size, embed_size, nhidden, nlayers, model=model, cuda=cuda)
lm = lm.to(device)
try:
    losses = lm.fit(corpus, epochs, lookahead)
except KeyboardInterrupt:
    print("Interrupting training.")

if not os.path.exists(metrics_dir):
    os.makedirs(metrics_dir)

lm.metrics.save()
sentence = lm.generate("Make America", corpus)
print(sentence)

try:
  torch.save(lm, model_out)
except:
  print("Torch failed to serialize the model and I don't know why...") 
