import torch
import torch.nn as nn
import torch.nn.functional as F
import data

class Model(nn.Module):
        
    def __init__(self, vocab_size, embed_size, nhidden, nlayers):
        super(Model, self).__init__()
        self.encoder = nn.Embedding(vocab_size, embed_size)
        self.rnn = nn.RNN(embed_size, nhidden, nlayers)
        self.decoder = nn.Linear(nhidden, vocab_size)

    def forward(self, x, h0):
        y = self.encoder(x)
        y = y.unsqueeze(0)
        y, h1 = self.rnn(y, h0)
        y = self.decoder(y)
        return y, h1

'''
Converts sequence to tensor.
'''
def seq2ix(seq, corpus):
    sequence = [corpus.word2ix[val] for val in seq.split()]
    return torch.tensor(sequence)

'''
Converts tensor output to actual words.
'''
def ix2word(data, corpus):
    to_word = lambda val, corpus: corpus.ix2word[val]
    return map(lambda x: to_word(x, corpus), data)


def batch(data, i, length):
    source = data[i:i+length]
    target = data[i+1:i+length+1].to(torch.long)
    return source, target

'''
Runs a single training epoch on the dataset.

Warning: One EPOCH took
CPU times: user 2min 45s, sys: 40.5 s, total: 3min 26s
Definitely needs to be trained on ICEHAMMER GPUs.
'''
def train(lm, corpus, seq_length, criterion=nn.CrossEntropyLoss(), lr=0.01, momentum=0.9, start = 0):
    lm.train()
    optimizer = torch.optim.SGD(lm.parameters(), lr=lr, momentum=momentum)
    total_loss = 0.
    data = corpus.data
    length = data.size(0)
    iterations = 1
    for i in range(start, length-seq_length, seq_length):
        lm.zero_grad()
        source, targets = batch(data, i, seq_length)
        source = source.to(torch.long)
        h0 = torch.zeros((nlayers, source.size(0), nhidden)) #TODO Come up with better initial hidden
        output, hidden = lm(source, h0)
        output = output.squeeze()
        loss = criterion(output, targets)
        total_loss += loss.item()
        iterations += 1
        loss.backward() 
        optimizer.step()
    return total_loss / iterations

def fit(lm, epochs, seq_length = 15):
    loss_history = []
    print("Running for {} epochs".format(epochs))
    for epoch in range(epochs):
        loss = train(lm, corpus, seq_length)
        print("Loss {}, Epoch {}".format(loss, epoch))
        loss_history.append(loss)
    return loss_history

def generate(lm, words, corpus):
    sentence = []
    for word in words.split():
        x0 = seq2ix(word, corpus)
        batch_size = x0.size(0)
        h0 = torch.zeros((nlayers, batch_size, nhidden))
        x1, h1 = lm(x0, h0)
        word = torch.multinomial(x1.div(1).exp(), 1).squeeze()
        sentence.extend(ix2word([word], corpus))
    return ' '.join(sentence)