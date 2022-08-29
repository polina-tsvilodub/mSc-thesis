import torch
import torch.nn as nn
from agents import resnet_encoder
from . import image_captioner
# from .agents import speaker
from transformers import TransfoXLLMHeadModel, TransfoXLTokenizer, TransfoXLConfig


class DriftMeter():
    """
    A class instantiating models and functions for computing language drift.
    """
    def __init__(self, structural_model, embed_size, vis_embed_size, hidden_size, vocab, semantic_encoder=None, semantic_decoder=None):
        """
        Initialize the object holding all functions and models for 
        computing the language drift metrics. Importantly, the models are only loaded once.

        Arguments:
        ---------
        semantic_encoder: path & name to weights of pretrained speaker MLP encoder 
        semantic_decoder: path & name to weights of pretrained LSTM 
        structural_model: string name of pretrained conditional LM from huggingface
        structural_tokenizer: string to pretrained huggingface tokenizer matching the LM 
        embed_size: int
        vis_embed_size: int
        vocab: length of vocab
        """
        super(DriftMeter, self).__init__()
        self.structural_model = TransfoXLLMHeadModel.from_pretrained(structural_model)
        self.structural_model.eval()
        self.tokenizer = TransfoXLTokenizer.from_pretrained(structural_model)
        self.decoder = semantic_decoder
        self.encoder = semantic_encoder
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.visual_embed_size = vis_embed_size
        self.vocab_len = vocab
        # instantiate models
        # ignore if they're not instantiated (e g when using in speaker pretraining)
        if semantic_decoder is not None:
            self.semantic_decoder = image_captioner.DecoderRNN(self.embed_size, self.hidden_size, self.vocab_len, self.visual_embed_size) #image_captioner.ImageCaptioner(self.embed_size, self.hidden_size, self.vocab_len) # this is a 1-image conditioned one now (with prepenading of embedding)
            try:
                self.semantic_decoder.load_state_dict(torch.load(self.decoder))
            except:
                self.semantic_decoder.load_state_dict(torch.load(self.decoder, map_location=torch.device('cpu')))
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.semantic_decoder.to(device)
            self.semantic_decoder.eval()
            self.hidden_state = self.semantic_decoder.init_hidden(64)
        # softmax for computing the probabilities over the scores
        self.softmax = nn.Softmax(dim=-1)
        


    # language drift metric (a)
    def compute_discrete_overlap(self, generated_caps, target_caps, distractor_caps):
        """
        Compute an overlap score over the generated caption with the two ground truth captions.
        This score is an attempt to capture language drift, while being agnostic towards compositional alternations.
        
        For batched input, expected input shape is (batch_size, caption_len), output is (batch_size,)
        generated_cap: index lists
            Caption generated by the speaker.
        target_cap:
            Ground truth caption for the target image.
        distractor_cap:
            Ground truth caption for the distractor image.
        """
        overlap_scores = []
        for j in range(generated_caps.shape[0]):
            target_score_list = [i in target_caps[j] for i in generated_caps[j]]
            distractor_score_list = [i in distractor_caps[j] for i in generated_caps[j]]
            overlap_score = sum(target_score_list) - sum(distractor_score_list)
            overlap_scores.append(overlap_score)
        
        return sum(overlap_scores)/len(overlap_scores)

    # metric (b)

    def compute_cont_overlap(self, generated_caps, target_caps, distractor_caps):
        """
        Compute an overlap score over the generated caption with the two ground truth captions.
        This score is an attempt to capture language drift, while being agnostic towards compositional alternations.
        Lower values indicate that target and prediction are closer than distractor and prediction.

        generated_cap: embeddings tensors (batch_size, len_caption, embed_size)
            Caption generated by the speaker.
        target_cap:
            Ground truth caption for the target image.
        distractor_cap:
            Ground truth caption for the distractor image.
            
        Returns:
            overlap_scor (batch_size,)
                tensor of cosine similarity scores by-caption.
        """
        target_dist = nn.functional.cosine_similarity(generated_caps, target_caps, dim=-1) # elementwise, then take average 
        distractor_dist = nn.functional.cosine_similarity(generated_caps, distractor_caps, dim=-1)
        # the mean computation makes it batch-level
        overlap_score = target_dist.mean(dim=-1) - distractor_dist.mean(dim=-1)
        return overlap_score.item() 

    # old metric
    def semantic_drift(self, caption, image, return_batch_average=True):
        """
        P(caption|image) under image caption model pretrained on one image only.
        
        image: (batch_size, 3, 224, 224)
        caption: (batch_size, caption_len)
        return_batch_average: whether to average over the batch (needs to be False for fixed listener).
        
        Returns:
        -------
            prob: (batch_size,)
                Tensor of conditional log probabilities measuring the semantic drift. 
        """
        # load pretrained models
        sent_probs = []
        with torch.no_grad():   
            # embed image
            # pass image, embedded caption through lstm
            if len(caption.shape) < 2:
                batch_size = 1
                image = image.unsqueeze(0)
                caption = caption.unsqueeze(0)
            else:
                batch_size = caption.shape[0]
            hidden = self.hidden_state 
            scores, h = self.semantic_decoder(image, caption, hidden) 
            # retrieve log probs of the target tokens (probs at given indices) 
            scores_prob = self.softmax(scores) 
            max_preds, max_inds = torch.max(scores_prob, dim = -1)
            # exclude START and END tokens
            sent_probs = []
            for num in list(range(batch_size)):
                sent_probs.append(
                    torch.stack(
                        [scores_prob[num][i][j] for i, j in enumerate(caption[num][:-1])] # cut off hypothesized end token; start token wasnt appended during generation by design  
                    )
                ) 

            # compute log probability of the sentence
            prob = torch.log(torch.stack(sent_probs)).sum(dim=1)
        # return softmax output for usage in KL divergence computation
        if return_batch_average:
            prob = prob.mean()
        return prob, scores_prob

    def structural_drift(self, caption):
        """
        P(caption) under some pretrained language model. 
        
        Caption needs to be natural language str.
        """
        inputs = self.tokenizer(caption, return_tensors="pt")
        with torch.no_grad():
            # pass labels in order to get neg LL estimates of the inputs as the loss
            outputs = self.structural_model(**inputs, labels = inputs["input_ids"])
            neg_ll = outputs[0]
        # compute sentence-level LL
        sent_ll = -neg_ll.sum(-1)
        # compute batch-level drift
        batch_ll = sent_ll.mean()
        return batch_ll

    def image_similarity(self, img1, img2):
        """
        Computes cosine similarity of image ResNet feature vectors.
        Averaged over batch.

        img1: torch.tensor(batch_size, 2048)
        img2: torch.tensor(batch_size, 2048)

        Returns:
        -------
        img_sim: float
            Pairwise cosine similarity between images. 
        """    

        cos_sim = nn.functional.cosine_similarity(img1, img2, dim=-1) 
        if len(img1.shape) > 1:
            cos_sim_batch = cos_sim.mean(dim=-1)
        else:
            cos_sim_batch = cos_sim
        return cos_sim_batch.item()