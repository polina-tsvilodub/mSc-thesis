import sys
import os
import math
import torch
import numpy as np
import pandas as pd
from random import shuffle
from . import early_stopping
import torch.nn as nn 
from drift_metrics import metrics
import random

def validate_model(
    data_loader_val,
    decoder,
):
    """
    Utility function for computing the validation performance of the speaker model while pre-training.
    
    Parameters:
    ---------
    data_loader_val:
        Validation data loader for images not used during training.
    decoder: 
        Trained decoder RNN to be evaluated.

    Returns:
    -------
    val_loss: float
        Average validation loss over the evaluation images.
    """
    # init params
    val_running_loss = 0.0
    counter = 0
    embedded_imgs = torch.load("./notebooks/resnet_pretrain_embedded_val_imgs.pt")
    total_steps = math.floor(len(data_loader_val.dataset.caption_lengths) / data_loader_val.batch_sampler.batch_size)
    print("Val total steps: ", total_steps+1)
    criterion = nn.CrossEntropyLoss().cuda() if torch.cuda.is_available() else nn.CrossEntropyLoss()

    decoder.eval()

    # evaluate model on the batch
    with torch.no_grad():
        print("Validating the model...")
        for i in range(1, total_steps+1): 
             # get val indices
            indices = data_loader_val.dataset.get_func_train_indices(i)

            new_sampler = torch.utils.data.sampler.SubsetRandomSampler(indices=indices)
            data_loader_val.batch_sampler.sampler = new_sampler

            counter += 1
            # Obtain the batch.
            targets, distractors, target_features, distractor_features, target_captions, distractor_captions = next(iter(data_loader_val))

            # Move batch of images and captions to GPU if CUDA is available.
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            targets = targets.to(device)
            distractors = distractors.to(device)
            target_captions = target_captions.to(device)

            # compute val predictions and loss
            both_images = torch.cat((target_features, distractor_features), dim=-1)
            outputs = decoder(both_images, target_captions)
            
            # The size of the vocabulary.
            vocab_size = len(data_loader_val.dataset.vocab)

            # Calculate the batch loss.
            loss = criterion(outputs.contiguous().view(-1, vocab_size), target_captions[:, 1:].reshape(-1))  

            val_running_loss += loss.item()
            
        
        val_loss = val_running_loss / counter
        print("Val loss: ", val_loss)
            
        return val_loss


def pretrain_speaker(
    log_file,
    num_epochs,
    total_steps,
    data_loader,
    data_loader_val, 
    decoder,
    params,
    criterion,
    optimizer,
    weights_path,
    print_every,
    save_every,
):
    """
    Training loop for pretraining the speaker model.

    Args:
    -----
        log_file: str
            Path to file for logging loss and perplexity.
        num_epochs: int
            Number of epochs to train for.
        total_steps: int
            Number of steps per epoch, calculated from batch size as dataset size.
        data_loader: torch.DataLoader
            Data loader batching the img caption pairs.
        data_loader_val: torch.DataLoader
            Data loader batching the img caption pairs for validation (separate val images).
        decoder: LSTM
            LSTM decoder instance.
        params: list
            Trainable model parameters.
        criterion: nn.Loss
            Loss to be applied.
        optimizer: nn.Optimizer
            Optimizer.
        weights_path: str
            Path to directory for saving mode weights.
        print_every: int
            Frequency of printing loss and perplecity (steps).
        save_every: int
            Frequency of saving model weights (epochs).
    """

    # Open the training log file.
    f = open(log_file, 'w')
    csv_out = "../../data/final/losses_final_pretrained_speaker_coco_4000vocab_tf_desc05_padding_pureDecoding_"
    val_csv_out = "../../data/final/val_final_pretrained_speaker_coco_4000vocab_tf_desc05_padding_pureDecoding_"
    csv_metrics = "../../data/final/metrics_final_pretrained_speaker_coco_4000vocab_tf_desc05_padding_pureDecoding_"
    
    speaker_losses=[]
    perplexities = []
    image_similarities = []
    steps = []
    val_losses, val_steps = [], []
    eval_steps = []
    structural_drifts_true = []
    structural_drifts = []
    semantic_drifts = []
    semantic_drifts_true = []
    discrete_overlaps = []
    cont_overlaps = []
    train_type = []
    forcing_rate = []
    sanity_check_inds = []

    # init the early stopper
    early_stopper = early_stopping.EarlyStopping(patience=3, min_delta=0.03)

    # teacher forcing schedule
    use_teacher_forcing_rate = 1
    # the value of the inverse sigmoid below is subject to hyperparam tuning as well
    # the higher the number, the longer teacher focing will be used
    k = 150

    # init drift meter for tracking structural drift for reference
    drift_meter = metrics.DriftMeter(
        # semantic_encoder="models/encoder-earlystoppiing-4_semantic-drift.pkl", 
        # semantic_decoder="models/decoder-3dshapes-512dim-47vocab-rs1234-wEmb-cont-5.pkl", 
        structural_model="transfo-xl-wt103",  
        embed_size=512, 
        vis_embed_size=512, 
        hidden_size=512,
        vocab=len(data_loader.dataset.vocab)
    )
    softmax = nn.Softmax(dim=-1)
    # create a list of "steps" which can be shuffled as a proxy for shuffling the indices of the images used for batching
    steps_nums = list(range(1, total_steps+1))

    for epoch in range(1, num_epochs+1):
        shuffle(steps_nums)
        for i, i_step in enumerate(steps_nums):
            decoder.train()
            # set models into training mode
            hidden = decoder.init_hidden(batch_size=data_loader.batch_sampler.batch_size)
                    
            # Randomly sample a caption length, and sample indices with that length.
            indices = data_loader.dataset.get_func_train_indices(i_step)
            sanity_check_inds.append(indices)
            # Create and assign a batch sampler to retrieve a batch with the sampled indices.
            new_sampler = torch.utils.data.sampler.SubsetRandomSampler(indices=indices)
            data_loader.batch_sampler.sampler = new_sampler
            
            # Obtain the batch.
            targets, distractors, target_features, distractor_features, target_captions, distractor_captions = next(iter(data_loader)) 
                
            # Move batch of images and captions to GPU if CUDA is available.
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            targets = targets.to(device)
            distractors = distractors.to(device)
            target_captions = target_captions.to(device)
            
            # Zero the gradients (reset).
            decoder.zero_grad()
            
            # concat image features
            both_images = torch.cat((target_features.unsqueeze(1), distractor_features.unsqueeze(1)), dim=1) 
            
            ### scheduled sampling 
            if use_teacher_forcing_rate == "scheduled":
                p = (k / (k + np.exp( ((epoch - 1) * total_steps + i_step) / k)))
                print("P : ", p)
                forcing_rate.append(p)
                train_type.append("scheduled_sampling")
                # init
                indices = []
                outputs = [] # for structural loss computation
                init_hiddens = decoder.init_hidden(data_loader.batch_sampler.batch_size) # Get initial hidden state of the LSTM
                
                # create initial caption input: "START"
                cat_samples = torch.tensor([0, 0]).repeat(data_loader.batch_sampler.batch_size, 1)
                for i in range(target_captions.shape[-1]-1):
                    out, hidden_state = decoder.forward(both_images, cat_samples, init_hiddens)
                    # for exp sampling
                    # out = out ** 5
                    outputs.append(out)
                    probs = softmax(out)

                    # flip coin which token to use
                    if np.random.choice([True, False], 1, p = [p, 1-p]):
                        
                        cat_samples = target_captions[:, i+1].unsqueeze(1) # i + 1 to skip start token 
                    # auto-regression option
                    else:
                       
                        # do pure sampling
                        # cat_dist = torch.distributions.categorical.Categorical(probs)
                        # greedy
                        _, cat_samples = torch.max(probs, dim = -1) #cat_dist.sample()
                        
                    indices.append(cat_samples)
                    cat_samples = torch.cat((cat_samples, cat_samples), dim = -1)

                outputs = torch.stack(outputs, dim=1).squeeze(2)

            #### classic teacher forcing 
            else:
                if random.random() < use_teacher_forcing_rate:
                    # print("Using teacher forcing")
                    train_type.append("teacher_forcing")
                    forcing_rate.append(use_teacher_forcing_rate)
                    outputs, hidden = decoder(both_images, target_captions, hidden)
                else:
                    # print("using self-regression")
                    forcing_rate.append(use_teacher_forcing_rate)
                    train_type.append("auto_regression")
                    decoder.eval()
                # print(f"doing self-regression for {target_captions.shape[1]} steps")
                # start_caption = torch.tensor([0, 0]).repeat(data_loader.batch_sampler.batch_size, 1)
                # for i in range(target_captions.shape[1] - 1):
                #     tokens_outputs, hidden = decoder(both_images, start_caption, hidden)
                #     print("SHAPE of LSTM out: ", outputs.shape)
                #     pred_prob, pred_ind = torch.max(outputs, dim = -1)
                #     # duplicate index for passing through the cutoff in the forward step
                #     start_caption = torch.cat((pred_ind, pred_ind), dim = -1)
                #     print("Duplicated predicted index: ", start_caption.shape)
                    max_seq_length = target_captions.shape[1]-1
                    captions_pred, log_probs, outputs, entropies = decoder.sample(
                        both_images, 
                        max_sequence_length=max_seq_length, 
                        decoding_strategy="pure"
                    )
            # The size of the vocabulary.
            vocab_size = len(data_loader.dataset.vocab)

            # Calculate the batch loss.
            loss = criterion(outputs.transpose(1,2), target_captions[:, 1:]) 
            # print("Loss: ", loss)
            # Backward pass.
            loss.backward()
            
            # Update the parameters in the optimizer.
            optimizer.step()
                
            # Get training statistics.
            stats = 'Epoch [%d/%d], Step [%d/%d], Loss: %.4f, Perplexity: %5.4f' % (epoch, num_epochs, i_step, total_steps, loss.item(), np.exp(loss.item()))
            
            speaker_losses.append(loss.item())
            steps.append(i_step)
            perplexities.append(torch.exp(loss).item())
        
            # Print training statistics (on same line).
            print('\r' + stats, end="")
            sys.stdout.flush()
            
            # Print training statistics to file.
            f.write(stats + '\n')
            f.flush()
            
            # Print training statistics (on different line).
            if i % 500 == 0:
                print('\r' + stats)
                decoder.eval()
                init_hidden = decoder.init_hidden(1)
                # also compute structural drift, for potential comparison to reference game setting
                for i in range(outputs.shape[0]): # iterate over sentences in batch
                    # structural drift under a pretrained LM
                    # decode caption to natural language for that
                    eval_steps.append(i_step)
                    outputs_ind = softmax(outputs[i])
                    max_probs, cat_samples = torch.max(outputs_ind, dim = -1)

                    nl_caption = [data_loader.dataset.vocab.idx2word[w.item()] for w in cat_samples]
                    nl_caption = " ".join(nl_caption)
                    structural_drift = drift_meter.structural_drift(nl_caption)
                    structural_drifts.append(structural_drift)
                    # also compute this for ground truth caption, as a reference value
                    # caption_ind = softmax(targets_captions[i])
                    nl_true_caption = [data_loader.dataset.vocab.idx2word[w.item()] for w in target_captions[i]]
                    nl_true_caption = " ".join(nl_true_caption)
                    structural_drift_true = drift_meter.structural_drift(nl_true_caption)
                    structural_drifts_true.append(structural_drift_true)

                    
        # Save the weights.
        if epoch % save_every == 0:
            torch.save(decoder.state_dict(), weights_path + '_%d.pkl' % epoch ) # os.path.join('./models', 'decoder-coco-512dim-teacher_forcing_scheduled_desc_05_byEp_pureDecoding_padded_indFixed-%d.pkl' % epoch)
            
            torch.save(sanity_check_inds, "../../data/final/sanity_checked_dataloader_indices.pt")
            use_teacher_forcing_rate = use_teacher_forcing_rate * 0.5
            # compute validation loss
            with torch.no_grad():
                val_loss = validate_model(
                    data_loader_val,
                    decoder,
                )     
                
                val_stats = 'Epoch [%d/%d], Step [%d/%d], Validation loss: %.4f' % (epoch, num_epochs, i_step, total_steps, val_loss)
                print(val_stats)
                f.write(val_stats + '\n')
                f.flush()   
                
                val_losses.append(val_loss)
                val_steps.append(i_step)
                # update stooper
                early_stopper(val_loss)
                # check if we want to break
                if early_stopper.early_stop:
                    # save the models
                    torch.save(decoder.state_dict(), weights_path + '_earlystopping_%d.pkl' % epoch)
                    print("Early stopping steps loop!")
                    break

        # save the training metrics
        df_out = pd.DataFrame({
            "steps": steps,
            "losses": speaker_losses,
            "perplexities": perplexities,
            "train_type": train_type,
            "forcing_rate": forcing_rate,
        })
        df_out.to_csv(csv_out + "epoch_" + str(epoch) + ".csv", index=False )

        df_val_out = pd.DataFrame({
            "steps": val_steps,
            "val_losses": val_losses
        })
        df_val_out.to_csv(val_csv_out + "epoch_" + str(epoch) + ".csv", index=False)

        metrics_out = pd.DataFrame({
            "steps": eval_steps,
            "structural_drift_true": structural_drifts_true,
            "structural_drift_pred": structural_drifts,
        })
        metrics_out.to_csv(csv_metrics + "epoch_" + str(epoch) + ".csv", index=False)

        # also break the epochs 
        if early_stopper.early_stop:
            print("Early stopping epochs loop!")
            break

    # Close the training log file.
    f.close()
