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
        # the tensor of shape (seq_len,) then unsqueeze() add a new dimension to the tensor (seq_len, 1), making it a column vector.
        
        div_term = torch.exp(torch.arange(0 , d_model , 2).float() * (-math.log(10000.0) / d_model) )
        
        # torch.arange(0, d_model, 2): if d_model is 512, this will produce a tensor of values: [0, 2, 4, ..., 510].
        
        # apply the sin to even positions 
        pe[: , ::2 ] = torch.sin(position * div_term) 