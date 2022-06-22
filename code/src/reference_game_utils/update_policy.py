import torch

def update_policy(rewards, log_probs, entropies, entropy_weight=0.1):
    """
    This function calculates the weight updates accoring to the REINFORCE rule.
    
    Args:
    ----
        rewards: list
            List of rewards of length batch_size
        log_probs: torch.tensor((batch_size,))
            Log probabilities of each word in each predicted sentence.
        entropies: torch.tensor(batch_size, caption_length)
            Tensor of sentence entropies
        entropy_weight: float
            Weight with which entropy regularization should be applied.    
    Returns:
    -----
        policy_gradient: torch.tensor
            Update to be applied together with the other loss components to the speaker parameters. 
    """

    policy_gradient = []
    sentence_prob = log_probs.sum(dim=1)
    sentence_entropies = entropies.sum(dim=1)
    for log_prob, Gt, h in zip(sentence_prob, rewards, sentence_entropies):
        # TODO double check sign
        # print("Regularization term: ", h, log_prob, Gt)

        policy_gradient.append(-(log_prob * Gt + entropy_weight * h))
    # here, we average over the batch to match the CCE operation for the joint loss
    policy_gradient = torch.stack(policy_gradient).mean()
    
    return policy_gradient

class MeanBaseline():
    """
    Helper for tracking and returning the mean baseline of past rewards, to be 
    discarded from loss.
    """
    def __init__(self):
        self.mean_baseline = 0.0
        self.n_steps = 0
    
    def update(self, reward):
        """
        Update current baseline based on new reward.
        """
        # TODO: assume i do this element-wise (sentence wise), and then batch-average at end
        self.mean_baseline += (reward - self.mean_baseline) / self.n_steps
        self.n_steps += 1
        
    def get(self):
        """
        Return current running mean baseline.
        """
        return self.mean_baseline    

def clean_sentence(captions, data_loader):
    """
    Helper for decoding captions to natural langauge for sanity checks.
    """

    clean_caps = []
    
    for idx in captions:
        list_string = []
#         print(idx)
        for i in idx:
#             print("i: ", i)
            try:
                list_string.append(data_loader.dataset.vocab.idx2word[i.item()])
            except ValueError:
                for y in i:
                    list_string.append(data_loader.dataset.vocab.idx2word[y.item()])
        list_string = list_string[:] # Discard <start> and <end> words
        sentence = ' '.join(list_string) # Convert list of string to full string
        sentence = sentence.capitalize()  # Capitalize the first letter of the first word
        clean_caps.append(sentence)
    return clean_caps