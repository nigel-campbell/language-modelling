import torch
from pprint import pprint

'''
Class for interfacing with corpus.

Heavily modified from this:

https://github.com/pytorch/examples/blob/master/word_language_model/data.py

Main difference: The original manually splits tests, train and validate rather
than just consuming the entire corpus. We will split test, train and validate
at runtime.
'''
class Corpus:
    
    def __init__(self, path, debug=False):
        self.word2ix = {}
        self.ix2word = []
        self.data = self._encode(self._read(path))

    def __len__(self):
        return len(self.ix2word)

    def _add(self, word):
        '''
        Adds word to dictionary.
        '''
        if not word in self.word2ix:
            self.ix2word.append(word)
            self.word2ix[word] = len(self.ix2word) - 1

    def _read(self, path):
        '''
        Reads corpus and initializes dictionary
        ''' 
        lines = []
        with open(path, 'r') as f:
            for line in f:
                words = line.split() + ['<eos>']
                for word in words:
                    self._add(word)
                lines += words
        return lines

    def _encode(self, lines):
        '''
        Properly encodes corpus into torch tensor.
        '''
        num_tokens = len(lines)
        ids = torch.Tensor(num_tokens)
        token = 0
        for word in lines:
            ids[token] = self.word2ix[word]
            token += 1 
        return ids

def seq_to_tensor(sentence, word_to_ix):
    '''
    Quick function to convert to a sequence
    into a tensor representation (given word_to_ix)
    '''
    idx = [word_to_ix[word] for word in sentence.split()]
    return torch.tensor(idx)


