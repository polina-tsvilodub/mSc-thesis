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

    csv_out = "functional_training_losses_token0_noEnc_vocab4000_metrics_"
    csv_metrics = "functional_language_drift_metrics_train_"

    speaker_losses_structural = []
    speaker_losses_functional = []
    listener_losses = []
    perplexities = []
    steps = []
    accuracies = []
    # TODO add metrics
    eval_steps = []
    semantic_drifts = []
    structural_drifts = []

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

    # init the drift metrics class
    drift_meter = metrics.DriftMeter(
        semantic_encoder="models/encoder-earlystoppiing-4_semantic-drift.pkl", 
        semantic_decoder="models/decoder-earlystopping-4_semantic-drift.pkl", 
        structural_model="transfo-xl-wt103",  
        embed_size=1024, 
        vis_embed_size=1024, 
        hidden_size=512,
        # TODO this will have to be retrained
        vocab=6039#len(data_loader.dataset.vocab)
    )

    for epoch in range(1, num_epochs+1):
        
        for i_step in range(1, total_steps+1):
            # set mode of the models
            speaker_decoder.train()
            # speaker_encoder.train()
            listener_encoder.train()
            listener_rnn.train()

            # Randomly sample a caption length, and sample indices with that length.
            indices_pairs = data_loader.dataset.get_func_train_indices()
            
            # Create and assign a batch sampler to retrieve a target batch with the sampled indices.
            new_sampler_pairs = torch.utils.data.sampler.SubsetRandomSampler(indices=indices_pairs)
            
            data_loader.batch_sampler.sampler = new_sampler_pairs
            # Obtain the target batch.
            images1, images2, target_features, distractor_features, captions = next(iter(data_loader))
            # create target-distractor image tuples
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            captions = captions.to(device)    
            
           
            ############################
            # Zero the gradients (reset).
            speaker_decoder.zero_grad()
            listener_encoder.zero_grad()
            listener_rnn.zero_grad()
            
            ###### Pass the images through the speaker model.
            both_images = [target_features, distractor_features]
            # sample caption from speaker 
            # zip images and target indices such that we can input correct image into speaker
            speaker_features_batch = []
            speaker_raw_output = []
            # get predicted caption and its log probability
            captions_pred, log_probs, raw_outputs, entropies = speaker_decoder.sample(both_images, max_sequence_length=captions.shape[1]-1)
            
            # transform predicted word indices to tensor
            # print("Raw indices grad: ", captions_pred.grad_fn)  
            # print("Log probs grad: ", log_probs.grad_fn)
            # print("Raw out grad: ", raw_outputs.grad_fn)
            #######
            # CREATE TARGET DIST RANDOM PAIRS FOR THE LISTENER
            targets_list = []
            features1_list = []
            features2_list = []
            target_indices_listener = np.random.choice([0,1], size=captions.shape[0]).tolist()
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
            targets_list = torch.tensor(targets_list).to(device)
            #######    

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
            # if target index and output index match, 1, else -1
            
            accuracy = torch.sum(torch.eq(targets_list, predictions).to(torch.int64))/predictions.shape[0]
            accuracies.append(accuracy.item())
            rewards = [1 if x else -1 for x in torch.eq(targets_list, predictions).tolist()]
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
            # print("Loss L s grads: ", loss_structural.grad_fn.next_functions)
            speaker_loss =  lambda_s*loss_structural + rl_grads
            # print("Speaker LOSS grads ", speaker_loss.grad_fn )
            # print("Speaker LOSS grads ", speaker_loss.grad_fn.next_functions )
            
            print("L_s: ", loss_structural, " L_f: ", rl_grads)
            
            # listener loss
            listener_loss = criterion(scores, targets_list)
            
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
                for i in range(captions_pred.shape[0]): # iterate over sentences in batch
                    # semantic drift under pretrained captioning model
                    semantic_drift = drift_meter.semantic_drift(captions_pred[i].unsqueeze(0), images1[i].unsqueeze(0))
                    eval_steps.append(i_step)
                    semantic_drifts.append(semantic_drift.item())
                    # structural drift under a pretrained LM
                    # decode caption to natural language for that
                    nl_caption = [data_loader.dataset.vocab.idx2word[w.item()] for w in captions_pred[i]]
                    nl_caption = " ".join(nl_caption)
                    structural_drift = drift_meter.structural_drift(nl_caption)
                    structural_drifts.append(structural_drift.item())
        # Save the weights.
        if epoch % save_every == 0:
            torch.save(speaker_decoder.state_dict(), os.path.join('./models', 'speaker-decoder-noEnc-token0-vocab4000-metrics-%d.pkl' % epoch))
            # torch.save(speaker_encoder.state_dict(), os.path.join('./models', 'speaker-encoder-singleImgs-token0-vocab6000-%d.pkl' % epoch))
            torch.save(listener_rnn.state_dict(), os.path.join('./models', 'listener-rnn-noEnc-token0-vocab4000-metrics-%d.pkl' % epoch))
            torch.save(listener_encoder.state_dict(), os.path.join('./models', 'listener-encoder-noEnc-token0-vocab4000-metrics-%d.pkl' % epoch))
            
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
        metrics_out = pd.DataFrame({
            "steps": eval_steps,
            "structural_drift": strucutral_drifts,
            "semantic_drifts": semantic_drifts,
        })
        metrics_out.to_csv(csv_metrics + "epoch_" + str(epoch) + ".csv", index=False)
    # Close the training log file.
    f.close()
    pass