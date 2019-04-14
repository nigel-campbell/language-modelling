import torch.nn as nn

class Model(nn.Module):
        
    def __init__(self, vocab_size, embed_size):
        super(Model, self).__init__()
        self.encoder = nn.Embedding(vocab_size, embed_size)
        self.decoder = nn.Linear(embed_size, vocab_size)

    def forward(self, _input):
        output = self.encoder(_input)
        output = self.decoder(output)
        return output

def eval(source, model):
    model.eval() # Disables dropout

def train(model):
    model.train() # Enables dropout
