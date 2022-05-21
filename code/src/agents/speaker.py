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
        self.lstm = nn.LSTM(self.embed_size, self.hidden_size , self.num_layers, batch_first=True) # self.embed_size+
        self.linear = nn.Linear(hidden_size, self.vocabulary_size)
        self.project = nn.Linear(2048, self.visual_embed_size)
        self.embed.weight.data.uniform_(-0.1, 0.1)

        self.linear.weight.data.uniform_(-0.1, 0.1)
        self.linear.bias.data.fill_(0)
        # init a hidden state such that this can be used for pretraining or first sampling step
        # and there is a placeholder for passing in an explicit hidden state when sampling
        self.batch_size = batch_size
        self.hidden = self.init_hidden(self.batch_size)

    def init_hidden(self, batch_size):
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
        target_emb = self.project(features[0]).unsqueeze(1)
        dist_emb = self.project(features[1]).unsqueeze(1)
        img_features = torch.cat((target_emb, dist_emb), dim=-1) 
        # print("Ft cat shape: ", img_features.shape)
        embeddings = self.embed(captions)
        # print("Raw embs shape: ", embeddings.shape)
        # features = features.unsqueeze(1)
        # print("features: ", features[0].shape)
        # print("embs: ", embeddings.shape)
        # PREpend the feature embedding as additional context as first token, cut off END token        
        embeddings = torch.cat((img_features, embeddings[:, :-1,:]), dim=1) # features_reps, dim=-1
        # print("Concat emb shape: ", embeddings.shape)
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
        batch_size = inputs[0].shape[0] # batch_size is 1 at inference, inputs shape : (1, 1, embed_size)
        hidden = self.init_hidden(batch_size) # Get initial hidden state of the LSTM
        softmax = nn.Softmax(dim=-1)
        # create initial caption input: "START"
        caption = torch.tensor([[0]]).repeat(batch_size, 1)
        print('In start: ', caption.shape)
        # make initial forward step, get output of shape (batch_size, 1, vocab_size)
        # print("Self hidden before the first sampling forward step: ", self.hidden)
        init_hiddens = self.init_hidden(batch_size)
        ####
        # outsource first step bc of image projection
        out, hidden_state = self.forward(inputs, caption, init_hiddens)
        print("Out shape after feeding img and start: ", out.shape)
        raw_outputs.extend(out)
        probs = softmax(out)
        if self.training:
            print("Train mode success")
            cat_dist = torch.distributions.categorical.Categorical(probs)
            cat_samples = cat_dist.sample()
            entropy = cat_dist.entropy()
            entropies.append(entropy)
            log_p = cat_dist.log_prob(cat_samples)
        else:
            max_probs, cat_samples = torch.max(probs, dim = -1)
            log_p = torch.log(max_probs)

        log_probs.append(log_p)
        output.append(cat_samples)
        word_emb = self.embed(cat_samples)




        # inputs = self.project(inputs)
        # print("Out shape: ", out.shape)
        print("Self hidden AFTER the first sampling forward step: ", hidden_state[0].shape, hidden_state[1].shape)
        
        # while True:
        for i in range(max_sequence_length):
            # make forward pass, get output of shape (batch_size, 1, vocab_size)
            # out = self.forward(inputs, caption)
            # print("Ind embeddings shape: ", word_emb[0].shape)
            # word_emb = word_emb.unsqueeze(1)
            lstm_out, hidden_state = self.lstm(word_emb, hidden_state)
            print("LSTM out shape", lstm_out.shape)
            # print("Self hidden after an iter of sampling loop: ", hidden_state )
            out = self.linear(lstm_out)
            print("Out shape at end of iter: ", out.shape)

            # get and save probabilities and save raw outputs
            raw_outputs.extend(out)
            probs = softmax(out)
            ####
            if self.training:
                # try sampling from a categorical
                cat_dist = torch.distributions.categorical.Categorical(probs)
                cat_samples = cat_dist.sample()
                entropy = cat_dist.entropy()
                entropies.append(entropy)
                print("Cat samples: ", cat_samples.shape)
                log_p = cat_dist.log_prob(cat_samples)
            else: 
                # if in eval mode, take argmax
                max_probs, cat_samples = torch.max(probs, dim = -1)
                log_p = torch.log(max_probs)

            # print("Log P computed w built ins: ", log_p)
            log_probs.append(log_p)
            # print(cat_samples)
            ####
            # max_probs, max_inds = torch.max(probs, dim = -1)
            # print("max inds shape: ", max_inds.shape)
            # scores.append(max_probs)
            output.append(cat_samples)
            print("max inds cat 0: ", cat_samples[0])
            # if (torch.equal(max_inds, torch.ones((inputs.shape[0], 1), dtype=torch.int64))) or (len(output) == max_sequence_length):
            #     # We predicted the <end> word or reached max length, so there is no further prediction to do
            #     break
            # embed predicted tokens
            word_emb = self.embed(cat_samples)
            

        # while True:
        #     lstm_out, hidden = self.lstm(inputs, hidden) # lstm_out shape : (1, 1, hidden_size)
        #     outputs = self.linear(lstm_out)  # outputs shape : (1, 1, vocab_size)
        #     print("Outputs shape: ", outputs.shape)
        #     raw_outputs.extend(outputs)
        #     # get the log probs of the actions
        #     probs = softmax(outputs)
        #     max_probs, max_inds = torch.max(probs, dim=-1)
        #     print("Max_inds: ", max_inds)
        #     scores.append(max_probs)
            
        #     outputs = outputs.squeeze(1) # outputs shape : (1, vocab_size)
        #     _, max_indice = torch.max(outputs, dim=1) # predict the most likely next word, max_indice shape : (1)
        #     print("Max inidce: ", max_indice)
        #     output.append(max_indice)
        #     if (torch.equal(max_indice, torch.ones((inputs.shape[0], 1), dtype=torch.int64))) or (len(output) == max_sequence_length):
        #         # We predicted the <end> word or reached max length, so there is no further prediction to do
        #         break
            
        #     ## Prepare to embed the last predicted word to be the new input of the lstm
        #     word_emb = self.embed(max_indice) # inputs shape : (1, embed_size)
        #     inputs = word_emb.unsqueeze(1) # inputs shape : (1, 1, embed_size)
            

        print("Output: ", len(output))
        output = torch.stack(output, dim=-1).squeeze(1)
        print("stacked output : ", output.shape)
        # stack
        log_probs = torch.stack(log_probs, dim=1).squeeze(-1)
        print("Lof probs at end: ", log_probs.shape)
        entropies = torch.stack(entropies, dim=1).squeeze(-1)
        print("Entropies: ", entropies.shape)

        ####
        # get effective log prob and entropy values - the ones up to (including) END (word2idx = 1)  
        # mask positions after END - both entropy and log P should be 0 at those positions
        end_mask = output.size(-1) - (torch.eq(output, 1).to(torch.int64).cumsum(dim=-1) > 0).sum(dim=-1)
        print("end mask: ", end_mask)
        # include the END token
        end_inds = end_mask.add_(1).clamp_(max=output.size(-1)) # shape: (batch_size,)
        print("End inds: ", end_inds)
        for pos, i in enumerate(end_inds):  
            # zero out log Ps and entropies
            log_probs[pos, i:] = 0
            entropies[pos, i:] = 0
        ####
        print("Zeroed entropies and lo Ps: ", log_probs, entropies)

        
        
        # if len(output) < max_sequence_length:
        #     # get the embedding and softmax output for pad
        #     pad = torch.tensor([3]).repeat(batch_size)
        #     pad_input = self.embed(pad).unsqueeze(1)
        #     pad_lstm_input = torch.cat((inputs, pad_input), dim=-1)
        #     lstm_pad, _ = self.lstm(pad_lstm_input, hidden)
        #     pad_output = self.linear(lstm_pad)
            
        #     while len(output) < max_sequence_length:
        #         output.append(pad) # pad
        #         raw_outputs.extend(pad_output)
        
        print("Raw outputs before stack: ", len(raw_outputs))
        raw_outputs = torch.stack(raw_outputs, dim=1).view(batch_size, -1, self.vocabulary_size)
        print("stacked raw output: ", raw_outputs.shape)
        # output.requires_grad_ = True
        return output, log_probs, raw_outputs, entropies