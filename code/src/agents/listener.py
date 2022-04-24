import torch
import torch.nn as nn
import torchvision.models as models 


class ListenerEncoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        """
        Initialize the langauge module consisting of a one-layer LSTM, a dropout layer and 
        trainable embeddings. The image embedding is used as additional context at every step of the training 
        (prepended at the embedding beginning). 
        """
        super(ListenerEncoderRNN, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.embed_size= embed_size
        self.vocabulary_size = vocab_size
        # not sure if this will be pretrained
        self.embed = nn.Embedding(self.vocabulary_size, self.embed_size) # .from_pretrained(
#                             torch.FloatTensor(self.fasttext.vectors)
#                         )

        self.lstm = nn.LSTM(self.embed_size, self.hidden_size , self.num_layers, batch_first=True)
        # reshape into a vector of (1, 512), such that similarity with the image can be computed
        self.linear = nn.Linear(hidden_size, 1)
        self.embed.weight.data.uniform_(-0.1, 0.1)

    # how is the lstm initialized?
        self.linear.weight.data.uniform_(-0.1, 0.1)
        self.linear.bias.data.fill_(0)
    def init_hidden(self, batch_size):
        """ At the start of training, we need to initialize a hidden state;
        there will be none because the hidden state is formed based on previously seen data.
        So, this function defines a hidden state with all zeroes
        The axes semantics are (num_layers, batch_size, hidden_dim)
        """
        return (torch.zeros((1, batch_size, self.hidden_size), device=device), \
                torch.zeros((1, batch_size, self.hidden_size), device=device))
    
    def forward(self, captions):
        # initialize hidden layer
        # check if this isn't reinitializing the hidden state in the middle of the sequence
#         self.hidden = self.init_hidden(self.hidden_size)
        
        embeddings = self.embed(captions)
        print("LSTM encoder embeddings shape: ", embeddings.shape)
        # where is the previous hidden state coming from?
        hiddens, self.hidden = self.lstm(embeddings) # , self.hidden
        outputs = self.linear(hiddens)
        print("LSTM encoder outputs shape: ", outputs.shape)
        return outputs