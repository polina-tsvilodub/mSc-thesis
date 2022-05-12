import sys
import os
import math
import torch
import numpy as np
import pandas as pd
import torch.nn as nn 

from drift_metrics import metrics

def play_game(
    log_file,
    num_epochs,
    total_steps,
    data_loader,
    data_loader_val, 
    encoder,
    decoder,
    params,
    criterion,
    optimizer,
    weights_path,
    print_every,
    save_every,
):
    # Open the training log file.
    f = open(log_file, 'w')

    csv_out = "functional_training_losses_2imgs_"

    speaker_losses_structural = []
    speaker_losses_functional = []
    listener_losses = []
    perplexities = []
    steps = []
    accuracies = []

    for epoch in range(1, num_epochs+1):
        
        for i_step in range(1, total_step+1):
            
            # Randomly sample a caption length, and sample indices with that length.
            indices_pairs = data_loader_pairs.dataset.get_func_train_indices()
            
            # Create and assign a batch sampler to retrieve a target batch with the sampled indices.
            new_sampler_pairs = torch.utils.data.sampler.SubsetRandomSampler(indices=indices_pairs)
            
            data_loader_pairs.batch_sampler.sampler = new_sampler_pairs
            # Obtain the target batch.
            images1, images2, captions = next(iter(data_loader_pairs))
            # create target-distractor image tuples
            train_pairs = list(zip(images1, images2))
            captions = captions.to(device)
            #########################
            # assume train_pairs is just (img1, img2, cap_img1)
            # TODO make this batch-level
            targets_list = []
            target_indices = np.random.choice([0,1], size=captions.shape[0]).tolist()
            print("Sampled target indices: ", target_indices)
            for i, target in enumerate(target_indices):
                if target == 0:
                    listener_img = (train_pairs[i][0], train_pairs[i][1])    
                else:
                    listener_img = (train_pairs[i][1], train_pairs[i][0])
                train_pairs[i] = listener_img    
                # memorize the target index    
                targets_list.append(target)
            #######    
            
            # TODO
            targets_list = torch.tensor(targets_list).to(device)
            print("Tensor targets list: ", targets_list)
            ############################
            # Zero the gradients (reset).
            speaker_encoder.zero_grad()
            speaker_decoder.zero_grad()
            listener_encoder.zero_grad()
            listener_rnn.zero_grad()
            
            ###### Pass the targets through the speaker model.
            
            # sample caption from speaker 
            # zip images and target indices such that we can input correct image into speaker
            preds_out = []
            log_probs_batch = []
            speaker_features_batch = []
            speaker_raw_output = []
            # get predicted captions for each image in the batch (to be made more efficient)
                # TODO
                # can I only embed this by-image, or can I concatenate more efficiently?
                # is this even correct to just call the visual encoder twice? because of the learning of the Linear layer later?
            target_speaker_features = speaker_encoder(images1)
            distractor_speaker_features = speaker_encoder(images2)
                # get predicted caption and its log probability
            # concatenate the two images 
            feature_pairs = torch.cat((target_speaker_features, distractor_speaker_features), dim=-1)
            captions_pred, log_probs, raw_outputs = speaker_decoder.sample(feature_pairs.unsqueeze(1), max_sequence_length=captions[i].shape[0])

            # transform predicted word indices to tensor
            preds_out = torch.stack(captions_pred, dim=-1)#.squeeze(-1)    
            #######
            # pass images and generated message form speaker through listener
            hiddens_scores, hidden = listener_rnn(preds_out) #  captions_pred
            # TODO this will change for new sampler, it will just take images which is batch of tuples (img1, img2)
            predictions = listener_encoder(train_pairs, hidden.squeeze(0)) 
            # retrieve the index of the larger dot product
            predicted_max_dots, predicted_inds = torch.max(predictions, dim = 1)
            ######
            # RL step
    #         log_probs = torch.stack(log_probs_batch)
            # if target index and output index match, 1, else -1
            
            accuracy = torch.sum(torch.eq(targets_list, predicted_inds).to(torch.int64))/predicted_inds.shape[0]
            accuracies.append(accuracy.item())
            rewards = [1 if x else -1 for x in torch.eq(targets_list, predicted_inds).tolist()]
            # compute REINFORCE update
            rl_grads = update_policy(rewards,  log_probs)
            
            
            # Calculate the batch loss.
            
            # REINFORCE for functional part, applied to speaker LSTM weights (maybe also Linear ones)
            # cross entropy for Listener
            # and also cross entropy for Speaker params, optimizing against target caption of the target image
            # (last implemented just like for pretraining), this is the structural loss component
            
            # combine structural loss and functional loss for the speaker # torch.stack(speaker_raw_output)
            loss_structural = criterion(torch.stack(raw_outputs).contiguous().view(-1, vocab_size), captions.reshape(-1)) 
            speaker_loss =  lambda_s*loss_structural + rl_grads
            
            print("L_s: ", loss_structural, " L_f: ", rl_grads)
            
            # listener loss
            listener_loss = criterion(predictions, targets_list)
            
            
            # Backward pass.
            speaker_loss.backward(retain_graph=True)
            listener_loss.backward(retain_graph=True)
            
            # Update the parameters in the respective optimizer.
            speaker_optimizer.step()
            listener_optimizer.step()
            
            # Get training statistics.
            # perplexity computation questionable
            stats = 'Epoch [%d/%d], Step [%d/%d], Speaker loss: %.4f, Listener loss: %.4f, Perplexity: %5.4f' % (epoch, num_epochs, i_step, total_step, speaker_loss.item(), listener_loss.item(), torch.exp(speaker_loss))
            
            speaker_losses_structural.append(loss_structural.item())
            speaker_losses_functional.append(rl_grads.item())
            listener_losses.append(listener_loss.item())
            perplexities.append(torch.exp(speaker_loss).item())
            steps.append(i_step)
            
            # Print training statistics (on same line).
            print('\r' + stats, end="")
            sys.stdout.flush()
            
            # Print training statistics to file.
            f.write(stats + '\n')
            f.flush()
            
            # Print training statistics (on different line).
            if i_step % print_every == 0:
                print('\r' + stats)
                # TODO double check

                # also compute the drift metrics during training to check the dynamics
                semantic_drift = metrics.semantic_drift(caption, image, visual_embed_size, embed_size, hidden_size, vocab_size)
                
        # Save the weights.
        if epoch % save_every == 0:
            torch.save(decoder.state_dict(), os.path.join('./models', 'decoder-2imgs-%d.pkl' % epoch))
            torch.save(encoder.state_dict(), os.path.join('./models', 'encoder-2imgs-%d.pkl' % epoch))
            
        # save the training metrics
        df_out = pd.DataFrame({
            "steps": steps,
            "speaker_s": speaker_losses_structural,
            "speaker_f": speaker_losses_functional,
            "listener": listener_losses,
            "perplexities": perplexities
        })
        df_out.to_csv(csv_out + "epoch_" + str(epoch) + ".csv", index=False )

    # Close the training log file.
    f.close()
    pass