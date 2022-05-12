import torch

def update_policy(rewards, log_probs):
    """
    This function calculates the weight updates accoring to the REINFORCE rule.
    
    Args:
    ----
        rewards: list
            List of rewards of length batch_size
        log_probs: torch.tensor((batch_size, caption_length))
            Log probabilities of each word in each predicted sentence.
    Returns:
    -----
        policy_gradient: torch.tensor
            Update to be applied together with the other loss components to the speaker parameters. 
    """

    policy_gradient = []
    sentence_prob = log_probs.sum(dim=1)
    for log_prob, Gt in zip(sentence_prob, rewards):
        policy_gradient.append(-log_prob * Gt)
    # here, we average over the batch to match the CCE operation for the joint loss
    policy_gradient = torch.stack(policy_gradient).mean()
    
    return policy_gradient