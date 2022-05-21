import sys
import os
import math
import torch
import numpy as np
import pandas as pd
import torch.nn as nn 

from drift_metrics import metrics
from . import update_policy

def play_game(
    log_file,
    num_epochs,
    total_steps,
    data_loader,
    data_loader_val, 
    # speaker_encoder,
    speaker_decoder,
    listener_encoder, 
    listener_rnn,
    criterion,
    weights_path,
    print_every,
    save_every,
):
    # Open the training log file.
    f = open(log_file, 'w')

    csv_out = "functional_training_losses_token0_noEnc_vocab4000_"

    speaker_losses_structural = []
    speaker_losses_functional = []
    listener_losses = []
    perplexities = []
    steps = []
    accuracies = []
    # TODO add metrics

    lambda_s = 0.1
    torch.autograd.set_detect_anomaly(True)

    # embedded_imgs = torch.load("COCO_train_ResNet_features_reshaped.pt")
    speaker_params = list(speaker_decoder.embed.parameters()) + list(speaker_decoder.lstm.parameters()) + list(speaker_decoder.linear.parameters()) + list(speaker_decoder.project.parameters()) 
    listener_params = list(listener_rnn.lstm.parameters()) + list(listener_encoder.embed.parameters()) 
    # print("Speaker decoder params: ", speaker_decoder.state_dict().keys())
    # print("speaker encoder params:", speaker_encoder.state_dict().keys())
    # print("Listener encoder params: ", listener_encoder.state_dict().keys())
    # print("Listener RNN : ", listener_rnn.state_dict().keys())
    # Define the optimizer.
    speaker_optimizer = torch.optim.Adam(speaker_params, lr=0.001, betas=(0.9, 0.999), eps=1e-08)
    listener_optimizer = torch.optim.Adam(listener_params, lr=0.001, betas=(0.9, 0.999), eps=1e-08)

    for epoch in range(1, num_epochs+1):
        
        for i_step in range(1, total_steps+1):
            # set mode of the models
            speaker_decoder.train()
            # speaker_encoder.train()
            listener_encoder.train()
            listener_rnn.train()

            # Randomly sample a caption length, and sample indices with that length.
            indices_pairs = data_loader.dataset.get_func_train_indices()
            # get the indices for retrieving the pretrained embeddings 
            # target_inds = torch.tensor([x[0] for x in indices_pairs]).long()
            # distractor_inds = torch.tensor([x[1] for x in indices_pairs]).long()
            # get the embeddings
            # target_features = torch.index_select(embedded_imgs, 0, target_inds).squeeze(1)
            # distractor_features = torch.index_select(embedded_imgs, 0, distractor_inds).squeeze(1)
            
            # Create and assign a batch sampler to retrieve a target batch with the sampled indices.
            new_sampler_pairs = torch.utils.data.sampler.SubsetRandomSampler(indices=indices_pairs)
            
            data_loader.batch_sampler.sampler = new_sampler_pairs
            # Obtain the target batch.
            images1, images2, target_features, distractor_features, captions = next(iter(data_loader))
            # create target-distractor image tuples
            print("traget features retrieved from dataloader: ", target_features.shape)
            # train_pairs = list(zip(images1, images2))
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            captions = captions.to(device)    
            
           
            # targets_list = torch.tensor(targets_list).to(device)
            # print("Tensor targets list: ", targets_list)
            ############################
            # Zero the gradients (reset).
            # speaker_encoder.zero_grad()
            speaker_decoder.zero_grad()
            listener_encoder.zero_grad()
            listener_rnn.zero_grad()
            
            ###### Pass the images through the speaker model.
            # project them with the linear layer
            # target_speaker_features = speaker_encoder(target_features)
            # distractor_speaker_features = speaker_encoder(distractor_features)
            # print("S encoder grads : ", target_speaker_features.grad_fn.next_functions)
            # concat image features AFTER projecting, target embedded first
            # both_images = torch.cat((target_speaker_features, distractor_speaker_features), dim=-1)
            both_images = [target_features, distractor_features]
            # sample caption from speaker 
            # zip images and target indices such that we can input correct image into speaker
            # preds_out = []
            # log_probs_batch = []
            speaker_features_batch = []
            speaker_raw_output = []
            # get predicted caption and its log probability
            # print("Max length for sampling: ", captions.shape[1])
            captions_pred, log_probs, raw_outputs, entropies = speaker_decoder.sample(both_images, max_sequence_length=captions.shape[1]-1)
            
            # transform predicted word indices to tensor
            # preds_out = torch.stack(captions_pred, dim=-1)#.squeeze(-1)  
            print(captions_pred[0])
            print("Raw indices grad: ", captions_pred.grad_fn)  
            print("Log probs grad: ", log_probs.grad_fn)
            print("Raw out grad: ", raw_outputs.grad_fn)
            #######
            # CREATE TARGET DIST RANDOM PAIRS FOR THE LISTENER
            targets_list = []
            features1_list = []
            features2_list = []
            target_indices_listener = np.random.choice([0,1], size=captions.shape[0]).tolist()
            # print("Sampled target indices: ", target_indices_listener)
            for i, target in enumerate(target_indices_listener):
                if target == 0:
                    features1_list.append(target_features[i])
                    features2_list.append(distractor_features[i])
                else:
                    features2_list.append(target_features[i])
                    features1_list.append(distractor_features[i])
  
                # memorize the target index    
                targets_list.append(target)
            features1 = torch.stack(features1_list)
            features2 = torch.stack(features2_list)
            # print("Optimized features1 lists: ", features1.shape)
            # print("Optimized features2 lists: ", features2.shape)
            targets_list = torch.tensor(targets_list).to(device)
            #######    


            print("Captions pred before L RNN; ", captions_pred.shape)

            # pass images and generated message form speaker through listener
            hiddens_scores, hidden = listener_rnn(captions_pred) #   preds_out
            # TODO check if these should be tuples or separate img1, img2 lists for using pre-extracted features
            predictions, scores = listener_encoder(features1, features2, hidden.squeeze(0)) # train_pairs
            # print("Listener encoder grads:", predictions.grad_fn.next_functions)
            # print("Listener encoder grads:", predictions.grad_fn.next_functions[0][0].next_functions)
            # print("Listener encoder grads:", predictions.grad_fn.next_functions[0][0].next_functions[0][0])
            # print("Listener encoder grads:", predictions.grad_fn.next_functions[0][0].next_functions[0][0].next_functions[0][0])
            # retrieve the index of the larger dot product
            # predicted_max_dots, predicted_inds = torch.max(predictions, dim = 1)
            ######
            # RL step
    #         log_probs = torch.stack(log_probs_batch)
            # if target index and output index match, 1, else -1
            
            accuracy = torch.sum(torch.eq(targets_list, predictions).to(torch.int64))/predictions.shape[0]
            print("Batch average accuracy: ", accuracy)
            accuracies.append(accuracy.item())
            rewards = [1 if x else -1 for x in torch.eq(targets_list, predictions).tolist()]
            print("Rewards: ", rewards)
            # compute REINFORCE update
            rl_grads = update_policy.update_policy(rewards, log_probs, entropies) # check if log probs need to be stacked first
            # print("RL rgrad: ", rl_grads.grad_fn.next_functions[0][0].next_functions[0][0].next_functions[0][0].next_functions[0][0].next_functions[0][0].next_functions)
            # compute entropies for each datapoint, to be weighted into loss
            
            # The size of the vocabulary.
            vocab_size = len(data_loader.dataset.vocab)
            
            # Calculate the batch loss.
            
            # REINFORCE for functional part, applied to speaker LSTM weights (maybe also Linear ones)
            # cross entropy for Listener
            # and also cross entropy for Speaker params, optimizing against target caption of the target image
            # (last implemented just like for pretraining), this is the structural loss component
            
            # combine structural loss and functional loss for the speaker # torch.stack(speaker_raw_output)
            loss_structural = criterion(raw_outputs.transpose(1,2), captions) 
            print("Loss L s grads: ", loss_structural.grad_fn.next_functions)
            speaker_loss =  lambda_s*loss_structural + rl_grads
            print("Speaker LOSS grads ", speaker_loss.grad_fn )
            print("Speaker LOSS grads ", speaker_loss.grad_fn.next_functions )
            
            print("L_s: ", loss_structural, " L_f: ", rl_grads)
            
            # listener loss
            listener_loss = criterion(scores, targets_list)
            print("L loss: ", listener_loss)
            
            # Backward pass.
            speaker_loss.backward(retain_graph=True)
            listener_loss.backward(retain_graph=True)
            
            # Update the parameters in the respective optimizer.
            speaker_optimizer.step()
            listener_optimizer.step()
            
            # Get training statistics.
            # perplexity computation questionable
            stats = 'Epoch [%d/%d], Step [%d/%d], Speaker loss: %.4f, Listener loss: %.4f, Perplexity: %5.4f, Accuracy: %.4f' % (epoch, num_epochs, i_step, total_steps, speaker_loss.item(), listener_loss.item(), torch.exp(loss_structural), accuracy.item())
            
            speaker_losses_structural.append(loss_structural.item())
            speaker_losses_functional.append(rl_grads.item())
            listener_losses.append(listener_loss.item())
            perplexities.append(torch.exp(loss_structural).item())
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
                # semantic_drift = metrics.semantic_drift(caption, image, visual_embed_size, embed_size, hidden_size, vocab_size)
                
        # Save the weights.
        if epoch % save_every == 0:
            torch.save(speaker_decoder.state_dict(), os.path.join('./models', 'speaker-decoder-noEnc-token0-vocab4000-%d.pkl' % epoch))
            # torch.save(speaker_encoder.state_dict(), os.path.join('./models', 'speaker-encoder-singleImgs-token0-vocab6000-%d.pkl' % epoch))
            torch.save(listener_rnn.state_dict(), os.path.join('./models', 'listener-rnn-noEnc-token0-vocab4000-%d.pkl' % epoch))
            torch.save(listener_encoder.state_dict(), os.path.join('./models', 'listener-encoder-noEnc-token0-vocab4000-%d.pkl' % epoch))
            
        # save the training metrics
        df_out = pd.DataFrame({
            "steps": steps,
            "speaker_s": speaker_losses_structural,
            "speaker_f": speaker_losses_functional,
            "listener": listener_losses,
            "perplexities": perplexities,
            "accuracies": accuracies
        })
        df_out.to_csv(csv_out + "epoch_" + str(epoch) + ".csv", index=False )

    # Close the training log file.
    f.close()
    pass