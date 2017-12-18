import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
import torch.autograd as autograd
import torch
import numpy as np

class BowModel(nn.Module):
    def __init__(self, emb_tensor):
        super(BowModel, self).__init__()
        n_embedding, self.dim = emb_tensor.size()
        self.embedding = nn.Embedding(n_embedding, self.dim, padding_idx=0)
        self.embedding.weight = Parameter(emb_tensor, requires_grad=False)
        self.lstm = nn.LSTM(input_size=self.dim, hidden_size=self.dim, num_layers=1, dropout=0.5)
        self.out = nn.Linear(self.dim, 2)
        self.hidden = self.init_hidden()

    def init_hidden(self):
        # Before we've done anything, we dont have any hidden state.
        # Refer to the Pytorch documentation to see exactly
        # why they have this dimensionality.
        # The axes semantics are (num_layers, minibatch_size, hidden_dim)
        return (autograd.Variable(torch.zeros(1, 1, self.dim)), autograd.Variable(torch.zeros(1, 1, self.dim)))
    
    def forward(self, input):
        '''
        input is a [batch_size, sentence_length] tensor with a list of token IDs
        '''
        embedded = self.embedding(input)
        # Here we take into account only the first word of the sentence
        # You should change it, e.g. by taking the average of the words of the sentence
        bow = embedded.mean(dim=1)
        #bow = embedded[:, 0]
        #return F.log_softmax(self.out(embedded)[1])
        bow, _ = self.lstm(bow, self.hidden)
        return F.log_softmax(self.out(bow[0, :, :]))
