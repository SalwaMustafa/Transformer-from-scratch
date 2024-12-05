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
        self.d_k = d_model // h
        # define the matrices by which we will multiply query, key and value and also the output matrix wo
        # weights in linear layer can be treated as a learnable matrix. 
        self.w_q = nn.Linear(d_model , d_model)
        self.w_k = nn.Linear(d_model , d_model)
        self.w_v = nn.Linear(d_model , d_model)
        self.w_o = nn.Linear(d_model , d_model)

        self.dropout = nn.Dropout(dropout)

    @staticmethod   # static method means that you can call this function without having an instance of this class (MultiHeadAttentionBlock.attention)
    def attention(query , key , value , mask , dropout: nn.Dropout):
        d_k = query.shape[-1]
        attention_scores = (query @ key.transpose(-2 , -1)) / math.sqrt(d_k)
        # @ means matrix multiplication in pytorch
        # (-2 , -1) means tanspose the last two dimensions from (seq_len , d_k) to (d_k , seq_len)
        # before applied the softmax we have to do the mask cuz we want sofmax to replace the masked values with zero
        if mask is not None :
            attention_scores.masked_fill(mask == 0 , -1e9) # that means that replace all the values for which the condition  is true with -1e9
        attention_scores = attention_scores.softmax(dim = -1) # (Batch , h , d_k , seq_len)
        if dropout is not None :
            attention_scores = dropout(attention_scores)
        
        return (attention_scores @ value) , attention_scores

    def forward(self , q , k , v , mask): 
        # if we want some words to not interact with future words we mask them or if we don't want the padding values to interact with other values
        query = self.w_q(q)  # shape (Batch , seq_len , d_model) ===>  shape (Batch , seq_len , d_model)
        key = self.w_k(k)    # shape (Batch , seq_len , d_model) ===>  shape (Batch , seq_len , d_model)
        value = self.w_v(v)  # shape (Batch , seq_len , d_model) ===>  shape (Batch , seq_len , d_model)

        # divide query, key and value into smaller matrices then give each small matrix to different head
        # shape (Batch , seq_len , d_model) ===> shape (Batch , seq_len , h , d_k) 
        # we want every head to be (seq_len , d_k) so we use the transpose ===> shape (Batch , h , seq_len , d_k)
        query = query.view(query.shape[0] , query.shape[1] , self.h , self.d_k).transpose(1,2)
        key = key.view(key.shape[0] , key.shape[1] , self.h , self.d_k).transpose(1,2)
        value = value.view(value.shape[0] , value.shape[1] , self.h , self.d_k).transpose(1,2)
        # view() function in PyTorch is used to reshape a tensor without changing its underlying data
        # Now, we have to calculate the attention using this formula ==> Attention(Q,K,V ) = softmax((Q * K**T) / sqrt(d_k)) * V
        x , self.attention_scores = MultiHeadAttentionBlock.attention(query , key , value , mask , self.dropout)

        x = x.transpose(1 , 2).contiguous().view(x.shape[0] , -1 , self.h * self.d_k) # shape (Batch , h , seq_len , d_k) ===> (Batch , seq_len , h , d_k) ===> (Batch , seq_len , d_model) 
        # we can not do a view direct so we use contiguous first
        # In PyTorch, after performing operations like transpose(), the tensor might no longer be stored in a contiguous block of memory
        # .contiguous() creates a new tensor that is stored in a contiguous block of memory, ensuring that operations like .view() can be safely applied
        
        return self.w_o(x) # shape (Batch , seq_len , d_model)  ===> (Batch , seq_len , d_model) 

class ResidualConnection(nn.Module):

    def __init__(self , dropout: float) -> None :
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNormalization()

    def forward(self , x , sublayer):
        return x + self.dropout(sublayer(self.norm(x)))  # Add & Norm


class EncoderBlock(nn.Module):

    def __init__(self, self_attention_block : MultiHeadAttentionBlock , feed_forward_block : FeedForward , dropout: float ) -> None:
        super().__init__()
        self.self_attention_block = self_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connection = nn.ModuleList([ResidualConnection(dropout) for _ in range(2)]) # in the encoder we need two Add & Norm layers
        # ModuleList is a way to organize a list of modules

    def forward(self , x , src_mask):
         # source mask is the mask we applied to the input of the encoder
         # we need a mask cuz we don't want the padding words to interact with other words
         # we have to make the skip connections 
         # for the first part of the encoder x is going to Add & Norm layer and at the same time  we need to apply the multi_head_attention on x
         x = self.residual_connection[0](x , lambda x: self.self_attention_block(x,x,x,src_mask)) # takes x and sublayer
         # residual_connection[0] represent the first skip connection 
         # x , x , x ==> query(q) , key(k) , value(v) cuz this is self attention (each word in one sentence is interact with other words of the same sentence )
         x = self.residual_connection[1](x , self.feed_forward_block)
         return x

class Encoder(nn.Module):

    def __init__(self, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers  # Nx
        self.norm = LayerNormalization()

    def forward(self , x , mask):
        for layer in self.layers :
            x = layer(x , mask)

        return self.norm(x)


# the output ambeddings are the same as the input embeddings so the class that we want to define is the same so we will just initialize it twice
# and the same does for the positional encoding 

class DecoderBlock(nn.Module):

    # masked multi_head_attention is self attention too cuz q , k , v are the same
    def __init__(self, self_attention_block : MultiHeadAttentionBlock , cross_attention_block : MultiHeadAttentionBlock , feed_forward_block : FeedForward , dropout : float) -> None:
        super().__init__()
        self.self_attention_block = self_attention_block
        self.cross_attention_block = cross_attention_block
        self.feed_forward_block = feed_forward_block

        # we have three residual connections 
        self.residual_connections = nn.ModuleList([ResidualConnection(dropout) for _ in range(3)])

    def forward(self , encoder_output , src_mask , tgt_mask):
        # src_mask is the mask applied to the encoder
        # tgt_mask is the mask applied to the decoder
        x = self.residual_connections[0]( x , lambda x : self.self_attention_block(x , x , x , tgt_mask))
        x = self.residual_connections[1]( x , lambda x : self.cross_attention_block(x , encoder_output , encoder_output , src_mask ))
        x = self.residual_connections[2]( x , self.feed_forward_block)

        return x

class Decoder(nn.Module):

    def __init__(self, layers : nn.ModuleList) -> None:
        super().__init__()   
        self.layers = layers # Nx
        self.norm = LayerNormalization()

    def forward(self , x , encoder_output , src_mask , tgt_mask):

        for layer in self.layers :
            x = layer(x , encoder_output , src_mask , tgt_mask)

        return self.norm(x)

# output of the decoder will be (Batch , seq_len , d_model)
# we want to map these words bach into the vocabulary
# the linear layer will convert the embedding into a position of the vocabulary 
# it's also called the projection layer because it's projecting the embedding into the vocabulary

class ProjectionLayer(nn.Module):

    def __init__(self, d_model : int , vocab_size : int ) -> None:
        super().__init__()
        self.proj = nn.Linear(d_model , vocab_size)

    def forward(self , x):
        # (Batch , seq_len , d_model) ===> (Batch , seq_len , vocab_size)
        return torch.log_softmax(self.proj(x) , dim = -1)

class Transformer(nn.Module):

    def __init__(self, encoder : Encoder , decoder : Decoder , src_embed : InputEmbeddings , tgt_embed : InputEmbeddings , src_pos : PositionalEncodeing , tgt_pos : PositionalEncodeing , projection_layer : ProjectionLayer) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.src_pos = src_pos
        self.tgt_pos = tgt_pos
        self.projection_layer = projection_layer

    def encode(self , src , src_mask):
        src = self.src_embed(src)
        src = self.src_pos(src)
        return self.encoder(src , src_mask)

    def decode(self , encoder_output , src_mask , tgt , tgt_mask):
        tgt = self.tgt_embed(tgt)
        tgt = self.tgt_pos(tgt)
        return self.decoder(tgt , encoder_output , src_mask , tgt_mask)

    def project(self , x):
        return self.projection_layer(x)

# we have to combine all these blocks together, we need one that given the hyper parameters of the transform 

def build_transformer(src_vocab_size : int , tgt_vocab_size : int , src_seq_len : int , tgt_seq_len : int , d_model : int = 512 , N : int = 6 , h : int = 8 , dropout : float = 0.1 , d_ff = 2048) -> Transformer :
    # create the embedding layers
    src_embed = InputEmbeddings(d_model , src_vocab_size)
    tgt_embed = InputEmbeddings(d_model , tgt_vocab_size)

    # create the positional encoding layers
    src_pos = PositionalEncodeing(d_model , src_seq_len , dropout)
    tgt_pos = PositionalEncodeing(d_model , tgt_seq_len , dropout)

    # create the encoder blocks 
    encoder_blocks = []
    for _ in range(N):
        encoder_self_attention_block = MultiHeadAttentionBlock(d_model , h , dropout)
        feed_forward_block = FeedForward(d_model , d_ff , dropout)
        encoder_block = EncoderBlock(encoder_self_attention_block , feed_forward_block , dropout)
        encoder_blocks.append(encoder_block)

    # create the decoder blocks
    decoder_blocks = []
    for _ in range(N):
        decoder_self_attention_block = MultiHeadAttentionBlock(d_model , h , dropout)
        decoder_cross_attention_block = MultiHeadAttentionBlock(d_model , h , dropout)
        feed_forward_block = FeedForward(d_model , d_ff , dropout)
        decoder_block = DecoderBlock(decoder_self_attention_block , decoder_cross_attention_block , feed_forward_block , dropout)
        decoder_blocks.append(decoder_block)

    # create the encoder and the decoder
    encoder = Encoder(nn.ModuleList(encoder_blocks))
    decoder = Decoder(nn.ModuleList(decoder_blocks))

    # create the projection layer 
    projection_layer = ProjectionLayer(d_model ,tgt_vocab_size)

    # create the transformer
    transformer = Transformer(encoder , decoder , src_embed , tgt_embed , src_pos , tgt_pos , projection_layer)

    # initialize the parameters to make the training faster so they don't just start with random value
    for p in transformer.parameters():
        if p.dim() > 1 :
            nn.init.xavier_uniform_(p)

    return transformer

