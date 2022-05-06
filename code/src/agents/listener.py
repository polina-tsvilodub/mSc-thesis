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
        self.embed = nn.Embedding(self.vocabulary_size, self.embed_size) 
        self.lstm = nn.LSTM(self.embed_size, self.hidden_size , self.num_layers, batch_first=True)
        self.embed.weight.data.uniform_(-0.1, 0.1)
        

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
        hiddens, self.hidden = self.lstm(embeddings)

        return hiddens, self.hidden[0] 

class ListenerEncoderCNN(EncoderCNN):
       
    def forward(self, image_pairs, caption): 
        """
        Performs forward pass through the listener ResNet 50 CNN.
        Computes the dot product between two images and the caption provided by the speaker.
        Outputs the index of the image which has the highest dot product with the caption - it is the predicted target.
        
        Args:
        ----
        images1: torch.tensor((batch_size, 3, 224, 224))
            List of images (potentially containing either targets or distractors).
        images2: torch.tensor((batch_size, 3, 224, 224))
            List of images (potentially containing either targets or distractors).
        caption: torch.tensor((batch_size, sentence_length, embed_size))
            Last hidden state of the RNN.
        Returns:
        ----
            indices: torch.tensor(batch_size)
                List of predicted target indices. 
            
        """
        # will be improved
        images1 = torch.stack([im[0] for im in image_pairs])
        images2 = torch.stack([im[1] for im in image_pairs])
        features1 = self.resnet(images1) 
        features2 = self.resnet(images2) 
        # reshape features to shape (batch_size, -1) - adapt to first dim
        features1 = features1.view(features1.size(0), -1)
        features1 = self.embed(features1)
        features2 = features2.view(features2.size(0), -1)
        features2 = self.embed(features2)
        # compute dot product between images and caption
        # compute mean over words as sentence embedding representation
        dot_products_1 = torch.bmm(features1.view(images1.size()[0], 1, features1.size()[1]),
                                   caption.view(images1.size()[0], features1.size()[1], 1))
        dot_products_2 = torch.bmm(features2.view(images2.size()[0], 1, features2.size()[1]),
                                   caption.view(images2.size()[0], features2.size()[1], 1))
        # compose targets and distractors dot products
        # stack into pairs, assuming dim=0 is the batch dimension
        pairs = torch.stack((dot_products_1, dot_products_2), dim=1) 
        pairs_flat = pairs.squeeze(-1).squeeze(-1)
        
        return pairs_flat        