import torch
import torch.nn as nn
import torchvision.models as models 

class ImageCaptioner(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        """
        Preatrainable Image captioning model used to compute the semsntic drift from Lazaridou et al.
        Initialize a one-layer LSTM and 
        trainable embeddings. 
        The image embedding is used as additional context at every step of the training 
        (prepended at the sentence beginning).
        
        Args:
        -----
            embed_size: int
                Dimensionality of trainable embeddings.
            hidden_size: int
                Hidden/ cell state dimensionality of the LSTM.
            vocab_size: int
                Length of vocabulary.
            num_layers: int
                Number of LST layers.
        """
        super(ImageCaptioner, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.embed_size= embed_size
        self.vocabulary_size = vocab_size
        self.lstm = nn.LSTM(self.embed_size, self.hidden_size , self.num_layers, batch_first=True)
        self.embed = nn.Embedding(self.vocabulary_size, self.embed_size) 
        self.linear = nn.Linear(hidden_size, self.vocabulary_size)
        self.embed.weight.data.uniform_(-0.1, 0.1)

        self.linear.weight.data.uniform_(-0.1, 0.1)
        self.linear.bias.data.fill_(0)

    def forward(self, features, captions):
        """
        Perform forward step through the LSTM.
        
        Args:
        -----
            features: torch.tensor((batch_size, embedd_size))
                Embeddings of images.
            captions: torch.tensor((batch_size, caption_length))
                Lists of indices representing tokens of each caption.
        Returns:
        ------
            outputs: torch.tensor((batch_size, caption_length, embedding_dim))
                Scores over vocabulary for each token in each caption.
        """
        
        embeddings = self.embed(captions)
        features = features.unsqueeze(1)
        # PREpend the feature embedding as additional context, cut off END token        
        embeddings = torch.cat((features, embeddings[:, :-1,:]), dim=1)
        hiddens, self.hidden = self.lstm(embeddings)
        
        outputs = self.linear(hiddens)
        return outputs
    