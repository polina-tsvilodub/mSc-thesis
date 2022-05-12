import torch
import torch.nn as nn
import torchvision.models as models 



class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, visual_embed_size, num_layers=1):
        """
        Initialize the langauge module consisting of a one-layer LSTM, a dropout layer and 
        trainable embeddings. The image embedding is used as additional context at every step of the training 
        (prepended at the embedding beginning). 

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
        super(DecoderRNN, self).__init__()
#         _ = "~/gensim-data/fasttext-wiki-news-subwords-300/fasttext-wiki-news-subwords-300.gz"
#         self.fasttext = gensim.models.KeyedVectors.load_word2vec_format(_)
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.embed_size= embed_size
        self.vocabulary_size = vocab_size
        self.visual_embed_size = visual_embed_size
        self.embed = nn.Embedding(self.vocabulary_size, self.embed_size) 
        self.lstm = nn.LSTM(self.embed_size + 2*self.visual_embed_size, self.hidden_size , self.num_layers, batch_first=True) # self.embed_size+
        self.linear = nn.Linear(hidden_size, self.vocabulary_size)
        self.embed.weight.data.uniform_(-0.1, 0.1)

        self.linear.weight.data.uniform_(-0.1, 0.1)
        self.linear.bias.data.fill_(0)

    def init_hidden(self, batch_size):
        """ At the start of training, we need to initialize a hidden state;
        there will be none because the hidden state is formed based on previously seen data.
        So, this function defines a hidden state with all zeroes
        The axes semantics are (num_layers, batch_size, hidden_dim)
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return (torch.zeros((1, batch_size, self.hidden_size), device=device), \
                torch.zeros((1, batch_size, self.hidden_size), device=device))
    
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
        features_reps = features.repeat(1, embeddings.shape[1]-1, 1)
        # PREpend the feature embedding as additional context AT EACH TIMESTEP, cut off END token        
        embeddings = torch.cat((features_reps, embeddings[:, :-1,:]), dim=-1) # features_reps, dim=-1
        hiddens, self.hidden = self.lstm(embeddings)
        
        outputs = self.linear(hiddens)
        return outputs
    
    def sample(self, inputs, max_sequence_length):
        """
        Function for sampling a caption during functional (reference game) training.
        Implements greedy sampling. Sampling stops when END token is sampled or when max_sequence_length is reached.
        Also returns the log probabilities of the action (the sampled caption) for REINFORCE.
        
        Args:
        ----
            inputs: torch.tensor(1, 1, embed_size)
                pre-processed image tensor.
            max_sequence_length: int
                Max length of sequence which the nodel should generate. 
        Returns:
        ------
            output: list
                predicted sentence (list of tensor ids). 
        """
        
        
        output = []
        raw_outputs = [] # for structural loss computation
        scores = []
        batch_size = inputs.shape[0] # batch_size is 1 at inference, inputs shape : (1, 1, embed_size)
        hidden = self.init_hidden(batch_size) # Get initial hidden state of the LSTM
        softmax = nn.Softmax(dim=-1)
        # embed the start token, repeat once for each item in the batch
        word_emb = self.embed(torch.tensor([0])).unsqueeze(0).repeat(inputs.shape[0], 1, 1)
        # below will be optimized
        while True:
            inputs_lstm = torch.cat((inputs, word_emb), dim=-1)
            lstm_out, hidden = self.lstm(inputs_lstm, hidden) # lstm_out shape : (1, 1, hidden_size)
            outputs = self.linear(lstm_out)  # outputs shape : (1, 1, vocab_size)
            raw_outputs.extend(outputs)
            # get the log probs of the actions
            probs = softmax(outputs)
            max_probs, max_inds = torch.max(probs, dim=-1)
            scores.append(max_probs)
            
            outputs = outputs.squeeze(1) # outputs shape : (1, vocab_size)
            _, max_indice = torch.max(outputs, dim=1) # predict the most likely next word, max_indice shape : (1)
            output.append(max_indice)
            if (torch.equal(max_indice, torch.ones((inputs.shape[0], 1), dtype=torch.int64))) or (len(output) == max_sequence_length):
                # We predicted the <end> word or reached max length, so there is no further prediction to do
                break
            
            ## Prepare to embed the last predicted word to be the new input of the lstm
            word_emb = self.embed(max_indice) # inputs shape : (1, embed_size)
            word_emb = word_emb.unsqueeze(1) # inputs shape : (1, 1, embed_size)
            
        # turn raw scores into log probabilities
        log_probs = torch.log(torch.stack(scores, dim=1))
        
        if len(output) < max_sequence_length:
            # get the embedding and softmax output for pad
            pad = torch.tensor([3]).repeat(batch_size)
            pad_input = self.embed(pad).unsqueeze(1)
            pad_lstm_input = torch.cat((inputs, pad_input), dim=-1)
            lstm_pad, _ = self.lstm(pad_lstm_input, hidden)
            pad_output = self.linear(lstm_pad)
            
            while len(output) < max_sequence_length:
                output.append(pad) # pad
                raw_outputs.extend(pad_output)
        
        return output, log_probs, raw_outputs