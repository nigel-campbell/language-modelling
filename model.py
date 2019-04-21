import torch
import torch.nn as nn
import torch.nn.functional as F
import data
from metrics import Metrics


class Model(nn.Module):
        
    def __init__(self, vocab_size, embed_size, nhidden, nlayers, model='LSTM',
            cuda=False, metrics=Metrics('metrics'), verbose=False):
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
        self.metrics = metrics

    def forward(self, x, h0):
        y = self.encoder(x)
        y = y.unsqueeze(0)
        y, h1 = self.rnn(y, h0)
        y = self.decoder(y)
        return y, h1


    def batch(self, data, i, length, lookahead):
        sources = []
        for j in range(lookahead+1):
            source = data[i+j:i+length+j].to(torch.long).to(self.device)
            sources.append(source)
        return sources
    
    '''
    Converts sequence to tensor.
    '''
    def seq2ix(self, seq, corpus):
        sequence = [corpus.word2ix[val] for val in seq.split()]
        return torch.tensor(sequence).to(self.device)

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
    def _train(self, corpus, data, seq_length, criterion=nn.CrossEntropyLoss(),
            lr=0.01, momentum=0.9, lookahead=2):
        self.train() # Sets the module in training mode.
        optimizer = torch.optim.SGD(self.parameters(), lr=lr, momentum=momentum)
        total_loss = 0.
        length = data.size(0)
        iterations = 1
        for i in range(0, length-seq_length, seq_length):
            self.zero_grad()
            sources = self.batch(data, i, seq_length, lookahead)
            for step in range(lookahead):
                source, target = sources[step], sources[step+1]
                hidden = self.init_hidden(source.size(0))
                output, hidden = self(source, hidden)
                output = output.squeeze()
                loss = criterion(output, target)
                total_loss += loss.item()
            iterations += 1
            loss.backward() 
            optimizer.step()
        return total_loss / iterations

    def fit(self, corpus, epochs, lookahead, seq_length = 15):
        print("Running for {} epochs at lookahead of {} steps".format(epochs,lookahead))
        data = corpus.data
        train_size = int(len(data) * 0.7)
        test_size = int(len(data) * 0.2)

        train = corpus.data[:train_size]
        test = corpus.data[train_size:train_size + test_size]
        validate = corpus.data[train_size + test_size:]
        for epoch in range(epochs):
            train_loss = self._train(corpus, train, seq_length, lookahead=lookahead)
            self.metrics.train_loss.append(train_loss)
            val_loss = self.evaluate(validate, corpus, seq_length, lookahead) 
            self.metrics.val_loss.append(val_loss)
            print("Train Loss {}, Validate Loss {} Epoch {}".format(train_loss,
                val_loss, epoch))
        return self.metrics.train_loss

    
    def generate(self, words, corpus, iterations=20, temperature=0.5):
        final_out = []
        for i in range(iterations):
            x0 = self.seq2ix(words, corpus)
            batch_size = x0.size(0)
            h0 = self.init_hidden(batch_size)
            x1, h1 = self(x0, h0)
            x1 = x1.squeeze().div(temperature).exp()
            values = torch.multinomial(x1, 1).squeeze()
            sentence = self.ix2word(values, corpus)
            words = ' '.join(sentence)
            final_out.extend(sentence)
        return ' '.join(final_out)
    
    '''
    Heavily borrows from PyTorch Language Model Example.
    Same as training except without gradient descent.
    '''
    def evaluate(self, data, corpus, seq_length, lookahead):
        self.eval()
        criterion=nn.CrossEntropyLoss()
        total_loss = 0.
        ntokens = len(corpus)
        iterations = 1
        length = data.size(0)
        with torch.no_grad():
            for i in range(0, length-seq_length, seq_length):
                sources = self.batch(data, i, seq_length, lookahead)
                for step in range(lookahead):
                    source, target = sources[step], sources[step+1]
                    hidden = self.init_hidden(source.size(0))
                    output, hidden = self(source, hidden)
                    output = output.squeeze()
                    loss = criterion(output, target)
                    total_loss += loss.item()
                iterations += 1
        return total_loss / iterations
        
