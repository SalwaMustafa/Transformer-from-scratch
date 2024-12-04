import torch 
import torch.nn as nn
import math


class InputEmbeddings(nn.Module):
    def __init__(self , d_model: int , vocab_size: int):
        super.__init__()
        self.d_model = d_model  # 512
        self.vocab_size = vocab_size

        # pytorch already provides with a layer that do what we want 
        # if it takes a number it will return the same vector every time
        
        self.embeddding = nn.Embedding(vocab_size,d_model)
    # From paper "In the embedding layers, we multiply those weights by the root of d_model."   
    def forward(self , x):
        return self.embedding(x) * math.sqrt(self.d_model) # these vectors are learned by the model
    

class PositionalEncodeing(nn.Module):
    def __init__(self , d_model: int , seq_len: int , dropout: float) -> None :
        #-> None: This is a type hint indicating that the method returns None, meaning it is not expected to return a value.
        super().__init__()
        self.d_model = d_model # 512
        self.seq_len = seq_len # the maximun length of the sentence cuz we want one vector for each position
        self.dropout = nn.Dropout(dropout)
        
        # we will build a matrix of shape (seq_len , d_model)
        pe = torch.zeroes(seq_len , d_model)
        
        # we will use log space instead of the formula that inside the sin and cos " if you apply the exponential then the log of something inside it
        # the result is the same number of these formula that inside the sin and cos but it's more numerical stable "
        
        # create a vector of shape(seq_len,1) that represent the position of the word inside the sentence
        
        position = torch.arange(0 , seq_len , dtype = torch.float).unsqueeze(1)
        
        # torch.arange(start, end): Generates a 1D tensor with values ranging from start to end - 1
        # the tensor of shape (seq_len,) then unsqueeze() add a new dimension to the tensor (seq_len, 1), making it a column vector
        
        div_term = torch.exp(torch.arange(0 , d_model , 2).float() * (-math.log(10000.0) / d_model) )
        
        # torch.arange(0, d_model, 2): if d_model is 512, this will produce a tensor of values: [0, 2, 4, ..., 510]
        
        # apply the sin to even positions 
        pe[: , ::2 ] = torch.sin(position * div_term) 
        # apply the cos to odd positions
        pe[: , 1::2] = torch.cos(position * div_term)
        # add batch dimension to this tensor cuz shape is (seq_len , d_model) but we will have batch of sentences
        pe = pe.unsqueeze(0) # (1 , seq_len , d_model)
        # register this tensor in the buffer of the module
        # Buffers are not considered model parameters (they won't be updated during training), but they are included in the module's state,
        # allowing them to be saved and loaded with the model
        self.register_buffer('pe' , pe)
    def forward(self , x):
        # add this positional encoding to every word inside the sentences
        x = x + (self.pe[: , :x.shape[1] , :]).require_grad(False)
        # : in the first dimension means take all positional encodings
        # :x.shape[1] in the second dimension means take the first x.shape[1] (seq_len) positional encodings from self.pe
        # : in the last dimension means all embedding dimensions
        return self.dropout(x)
    
class LayerNormalization(nn.Module):
    def __init__(self , eps: float = 10**-6) -> None : # eps refers to the epsilon in the denominator of the nomalization formula to avoid division by 0
        super().__init__()
        self.eps = eps 
        self.alpha = nn.parameter(torch.ones(1)) # multiplied 
        self.bias = nn.parameter(torch.zeros(1)) # added

    def forward(self , x):
        mean = x.mean(dim = -1 , keepdim = True) 
        std = x.std(dim = -1 , keepdim = True)
        return self.alpha * (x - mean)/(std + self.eps) + self.bias
    
class FeedForward(nn.Module):
    def __init__(self , d_model: int , d_ff: int , dropout: float ) -> None:
      super().__init__()  
      self.linear_1 = nn.Linear(d_model , d_ff) # W1 and B1
      self.dropout = nn.Dropout(dropout)
      self.linear_2 = nn.Linear(d_ff , d_model) # W2 and B2

    def forward(self , x):
        # x shape is (Batch , seq_len , d_model) and after the 1st layer it will be (Batch , seq_len , d_ff)
        # then after  the 2nd layer it will be   (Batch , seq_len , d_model) 
        return self.linear_2(self.dropout(torch.relu(self.linear_1(x))))
    

class MultiHeadAttentionBlock(nn.Module):
    def __init__(self , d_model: int , h: int , dropout: float ) -> None : # h is the number of heads
        super().__init__()
        self.d_model = d_model 
        self.h = h
        
        # we have to divide the embedding vector into h heads that means d_model shoud be divisible by h
        assert d_model % h == 0 , "d_model is not divisible by h" 
        # dk = dv = d_model / h
        self.dk = d_model // h
        # define the matrices by which we will multiply query, key and value and also the output matrix wo
        self.w_q = nn.Linear(d_model , d_model)
        self.w_k = nn.Linear(d_model , d_model)
        self.w_v = nn.Linear(d_model , d_model)
        self.w_o = nn.Linear(d_model , d_model)

        self.dropout = nn.Dropout(dropout)


    


