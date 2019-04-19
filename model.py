import torch
import torch.nn as nn
import torch.nn.functional as F
import data


def batch(data, i, length):
    source = data[i:i+length]
    target = data[i+1:i+length+1].to(torch.long)
    return source, target


class Model(nn.Module):
        
    def __init__(self, vocab_size, embed_size, nhidden, nlayers, model='LSTM', cuda=False):
        super(Model, self).__init__()
        self.encoder = nn.Embedding(vocab_size, embed_size)
        if model == 'LSTM':
            self.rnn = nn.LSTM(embed_size, nhidden, nlayers)
        else:
            self.rnn = nn.RNN(embed_size, nhidden, nlayers)
        self.decoder = nn.Linear(nhidden, vocab_size)
        
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.nhidden = nhidden
        self.nlayers = nlayers
        self.model = model
        self.device = torch.device("cpu" if not cuda else "cuda")

    def forward(self, x, h0):
        y = self.encoder(x)
        y = y.unsqueeze(0)
        y, h1 = self.rnn(y, h0)
        y = self.decoder(y)
        return y, h1

    '''
    Converts sequence to tensor.
    '''
    def seq2ix(self, seq, corpus):
        sequence = [corpus.word2ix[val] for val in seq.split()]
        return torch.tensor(sequence)

    '''
    Converts tensor output to actual words.
    '''
    def ix2word(self, data, corpus):
        to_word = lambda val, corpus: corpus.ix2word[val]
        return map(lambda x: to_word(x, corpus), data)

    def init_hidden(self, batch_size):
        h0 = torch.zeros((self.nlayers, batch_size, self.nhidden),
                device=self.device)
        if self.model == 'LSTM':
            return (h0, h0)
        return h0
    
    '''
    Runs a single training epoch on the dataset.

    Warning: One EPOCH took
    CPU times: user 2min 45s, sys: 40.5 s, total: 3min 26s
    Definitely needs to be trained on ICEHAMMER GPUs.
    '''
    def _train(self, corpus, seq_length, criterion=nn.CrossEntropyLoss(), lr=0.01, momentum=0.9, start = 0):
        self.train() # Sets the module in training mode.
        optimizer = torch.optim.SGD(self.parameters(), lr=lr, momentum=momentum)
        total_loss = 0.
        data = corpus.data
        length = data.size(0)
        iterations = 1
        for i in range(start, length-seq_length, seq_length):
            self.zero_grad()
            source, targets = batch(data, i, seq_length)
            source = source.to(torch.long).to(self.device)
            hidden = self.init_hidden(source.size(0)).to(self.device)
            output, hidden = self(source, hidden)
            output = output.squeeze()
            loss = criterion(output, targets)
            total_loss += loss.item()
            iterations += 1
            loss.backward() 
            optimizer.step()
        return total_loss / iterations

    def fit(self, corpus, epochs, seq_length = 15, cuda=False):
        loss_history = []
        print("Running for {} epochs".format(epochs))
        for epoch in range(epochs):
            loss = self._train(corpus, seq_length)
            print("Loss {}, Epoch {}".format(loss, epoch))
            loss_history.append(loss)
        return loss_history

    def generate(self, words, corpus, batch_size, temperature=1):
        sentence = []
        for word in words.split():
            x0 = seq2ix(word, corpus)
            batch_size = x0.size(0)
            h0 = torch.zeros((self.nlayers, batch_size, self.nhidden))
            x1, h1 = self(x0, h0)
            word = torch.multinomial(x1.div(temperature).exp(), 1).squeeze()
            sentence.extend(ix2word([word], corpus))
        return ' '.join(sentence)
