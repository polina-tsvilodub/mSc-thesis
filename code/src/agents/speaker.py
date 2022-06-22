import torch
import torch.nn as nn
import torchvision.models as models 



class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, visual_embed_size, batch_size=64, num_layers=1):
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
        self.project = nn.Linear(2048, self.visual_embed_size)
        self.embed.weight.data.uniform_(-0.1, 0.1)

        self.linear.weight.data.uniform_(-0.1, 0.1)
        self.linear.bias.data.fill_(0)
        # init a hidden state such that this can be used for pretraining or first sampling step
        # and there is a placeholder for passing in an explicit hidden state when sampling
        self.batch_size = batch_size
        self.hidden = self.init_hidden(self.batch_size)

    def init_hidden(self, batch_size): # TODO try init hidden with image representations
        """ At the start of training, we need to initialize a hidden state;
        there will be none because the hidden state is formed based on previously seen data.
        So, this function defines a hidden state with all zeroes
        The axes semantics are (num_layers, batch_size, hidden_dim)
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return (torch.randn((1, batch_size, self.hidden_size), device=device), \
                torch.randn((1, batch_size, self.hidden_size), device=device))
    
    def forward(self, features, captions, prev_hidden):
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
        # features of shape (batch_size, 2, 2048)
        image_emb = self.project(features) # image_emb should have shape (batch_size, 2, 512)
        # concatenate target and distractor embeddings
        img_features = torch.cat((image_emb[:, 0, :], image_emb[:, 1, :]), dim=-1).unsqueeze(1) 
        embeddings = self.embed(captions)
        # repeat image features such that they can be prepended to each token
        img_features_reps = img_features.repeat(1, embeddings.shape[1]-1, 1)
        # PREpend the feature embedding as additional context as first token, cut off END token        
        embeddings = torch.cat((img_features_reps, embeddings[:, :-1,:]), dim=-1) # features_reps, dim=-1
        hiddens, hidden_state = self.lstm(embeddings, prev_hidden)
        
        outputs = self.linear(hiddens)
        return outputs, hidden_state
    
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
        log_probs = []
        entropies = []
        batch_size = 64#inputs[0].shape[0] # batch_size is 1 at inference, inputs shape : (1, 1, embed_size)
        hidden = self.init_hidden(batch_size) # Get initial hidden state of the LSTM
        softmax = nn.Softmax(dim=-1)
        # create initial caption input: "START"
        caption = torch.tensor([0, 0]).repeat(64, 1) # two 0s since we cut off last token in forward step, so that we actually keep one
        # make initial forward step, get output of shape (batch_size, 1, vocab_size)
        init_hiddens = self.init_hidden(batch_size)
        ####
        # outsource first step bc of image projection
        out, hidden_state = self.forward(inputs, caption, init_hiddens)
        raw_outputs.append(out) #.extend(out)
        probs = softmax(out)
        if self.training:
            cat_dist = torch.distributions.categorical.Categorical(probs)
            cat_samples = cat_dist.sample()
            entropy = cat_dist.entropy()
            entropies.append(entropy)
            log_p = cat_dist.log_prob(cat_samples)
        else:
            max_probs, cat_samples = torch.max(probs, dim = -1)
            log_p = torch.log(max_probs)
        output.append(cat_samples)
        cat_samples = torch.cat((cat_samples, cat_samples), dim=-1)
        # print("Cat samples ", cat_samples)
        log_probs.append(log_p)
        # output.append(cat_samples)
        word_emb = self.embed(cat_samples)

        for i in range(max_sequence_length-1):
            # print("CAT SAMPLES AT BEGINNING OF SAMPLING LOOP ", cat_samples)
            out, hidden_state = self.forward(inputs, cat_samples, hidden_state)
            # lstm_out, hidden_state = self.lstm(word_emb, hidden_state)
            # print("Self out after an iter of sampling loop: ", out.shape )
            # out = self.linear(lstm_out)
            
            # get and save probabilities and save raw outputs
            raw_outputs.append(out)
            probs = softmax(out)
            ####
            if self.training:
                # try sampling from a categorical
                cat_dist = torch.distributions.categorical.Categorical(probs)
                cat_samples = cat_dist.sample()
                entropy = cat_dist.entropy()
                entropies.append(entropy)
                log_p = cat_dist.log_prob(cat_samples)
            else: 
                # if in eval mode, take argmax
                max_probs, cat_samples = torch.max(probs, dim = -1)
                log_p = torch.log(max_probs)
                entropy = -log_p * max_probs
                entropies.append(entropy)
            
            output.append(cat_samples)
            cat_samples = torch.cat((cat_samples, cat_samples), dim=-1)
            # print("Cat samples ", cat_samples)
            log_probs.append(log_p)
            ####
            # output.append(cat_samples)
            # embed predicted tokens
            word_emb = self.embed(cat_samples)
            
        # print("Len raw outputs ", len(raw_outputs))
        # print(raw_outputs[0].shape)
        # print(raw_outputs[1].shape)
        output = torch.stack(output, dim=-1).squeeze(1)
        # stack
        log_probs = torch.stack(log_probs, dim=1).squeeze(-1)
        entropies = torch.stack(entropies, dim=1).squeeze(-1)
        
        ####
        # get effective log prob and entropy values - the ones up to (including) END (word2idx = 1)  
        # mask positions after END - both entropy and log P should be 0 at those positions
        end_mask = output.size(-1) - (torch.eq(output, 1).to(torch.int64).cumsum(dim=-1) > 0).sum(dim=-1)
        # include the END token
        end_inds = end_mask.add_(1).clamp_(max=output.size(-1)) # shape: (batch_size,)
        for pos, i in enumerate(end_inds):  
            # print("end inds ", i)
            # zero out log Ps and entropies
            log_probs[pos, i:] = 0
            entropies[pos, i:] = 0
        ####
        # print('raw stacked outputs in sampling before applying view: ', torch.stack(raw_outputs, dim=1).shape)
        raw_outputs = torch.stack(raw_outputs, dim=1).squeeze(2)  #view(batch_size, -1, self.vocabulary_size)
        # print('raw outputs as received in training loop: ', raw_outputs.shape)
        return output, log_probs, raw_outputs, entropies