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

    csv_out = "functional_training_losses_wPretrainedFixed_3dshapes_ls02_"
    csv_metrics = "functional_training_wPretrainedFixed_3dshapes_language_drift_metrics_ls02_"
    # csv_out = "functional_training_losses_wPretrained_coco_ls02_orig_wSampling_"
    # csv_metrics = "functional_training_metrics_wPretrained_coco_ls02_orig_wSampling_"

    speaker_losses_structural = []
    speaker_losses_functional = []
    speaker_losses = []
    listener_losses = []
    perplexities = []
    steps = []
    accuracies = []
    image_similarities = []
    image_similarities_val = []
    kl_divs = []

    eval_steps = []
    structural_drifts_true = []
    structural_drifts = []
    semantic_drifts = []
    semantic_drifts_true = []
    discrete_overlaps = []
    cont_overlaps = []
    epochs_out = []

    lambda_s = 0.2
    kl_coeff = 0.1
    # torch.autograd.set_detect_anomaly(True)

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

    # LR scheduler
    # scheduler_s = torch.optim.lr_scheduler.StepLR(speaker_optimizer, step_size=1000, gamma=0.5)
    # scheduler_l = torch.optim.lr_scheduler.StepLR(listener_optimizer, step_size=1000, gamma=0.5)

    # init the drift metrics class
    drift_meter = metrics.DriftMeter(
        # semantic_decoder="models/decoder-3dshapes-512dim-47vocab-rs1234-wEmb-short-5.pkl",
        semantic_decoder="models/decoder-3dshapes-512dim-49vocab-rs1234-exh-3.pkl", 
        # semantic_decoder="models/decoder-noEnc-prepend-512dim-4000vocab-rs1234-wEmb-cont-7.pkl", 
        structural_model="transfo-xl-wt103",  
        embed_size=512, 
        vis_embed_size=512, 
        hidden_size=512,
        # TODO this will have to be retrained
        vocab=len(data_loader.dataset.vocab)
    )
    softmax = nn.Softmax(dim=-1)

    # mean reward baseline variance stabilisation
    mean_baseline = update_policy.MeanBaseline()

    for epoch in range(1, num_epochs+1):
        
        for i_step in range(1, total_steps+1):
            # set mode of the models
            speaker_decoder.train()
            # speaker_encoder.train()
            listener_encoder.train()
            listener_rnn.train()

            # Randomly sample a caption length, and sample indices with that length.
            indices_pairs = data_loader.dataset.get_func_train_indices()
            # print("inds ", indices_pairs)
            
            # Create and assign a batch sampler to retrieve a target batch with the sampled indices.
            new_sampler_pairs = torch.utils.data.sampler.SubsetRandomSampler(indices=indices_pairs)
            
            data_loader.batch_sampler.sampler = new_sampler_pairs
            # Obtain the target batch.
            images1, images2, target_features, distractor_features, captions, dist_captions = next(iter(data_loader))
            # create target-distractor image tuples
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            captions = captions.to(device)    
            
            # compute image similarities
            cos_sim_img = drift_meter.image_similarity(target_features, distractor_features)
            image_similarities.append(cos_sim_img)
            ############################
            # Zero the gradients (reset).
            speaker_decoder.zero_grad()
            listener_encoder.zero_grad()
            listener_rnn.zero_grad()
            
            ###### Pass the images through the speaker model.
            both_images = torch.cat((target_features.unsqueeze(1), distractor_features.unsqueeze(1)), dim=1)
            # sample caption from speaker 
            # zip images and target indices such that we can input correct image into speaker
            speaker_features_batch = []
            speaker_raw_output = []
            # get predicted caption and its log probability
            captions_pred, log_probs, raw_outputs, entropies = speaker_decoder.sample(both_images, max_sequence_length=captions.shape[1]-1)
            print("---- GROUND TRUTH CAPTIONS ---- ", update_policy.clean_sentence(captions, data_loader)[:2])
            print("---- PREDICTED CAPTIONS ---- ", update_policy.clean_sentence(captions_pred, data_loader)[:2])
            
            # print("Captions pred ", captions_pred)
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
            # print("captions_pred ", captions_pred)
            hiddens_scores, hidden = listener_rnn(captions_pred) #   preds_out
            # TODO check if these should be tuples or separate img1, img2 lists for using pre-extracted features
            features = torch.cat((features1.unsqueeze(1), features2.unsqueeze(1)), dim=1)
            # print("CONCAT FEATURES FOR LISTENER BEFORE PASSING INTO ENCODER ", features.shape)
            predictions, scores = listener_encoder(features, hidden.squeeze(0)) # train_pairs
        
            ######
            # RL step
            # if target index and output index match, 1, else -1
            
            accuracy = torch.sum(torch.eq(targets_list, predictions).to(torch.int64))/predictions.shape[0]
            accuracies.append(accuracy.item())
            rewards = [1 if x else -1 for x in torch.eq(targets_list, predictions).tolist()]
            #####
            # mean baseline
            # b = mean_baseline.get()
            # print("current Baseline before update: ", b)
            # print("rewards to be added to baseline comp: ", rewards)
            # mean_baseline.update(rewards) # TODO check batch dims and mea computation sensibility
            # rewards = rewards - b # TODO CHEEEECK 
            ####
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
            
            # combine structural loss and functional loss for the speaker 
            # compute distribution under pretrained model

            ###########
            # KL divergence 
            ###########
            # semantic_drift, pretrained_prob = drift_meter.semantic_drift(captions, both_images) 
            # # should probably also be averaged over batch for mean batch loss reduction
            # raw_outputs_probs = softmax(raw_outputs)
            # raw_outputs_dist = torch.distributions.categorical.Categorical(probs=raw_outputs_probs)
            # pretrained_prob_dist = torch.distributions.categorical.Categorical(probs=pretrained_prob)
            # # KL divs of shape (batch_size,)
            # # print("SHAPES OF PROB TENSORS")
            # # print(raw_outputs_probs.shape)
            # # print(pretrained_prob.shape)
            # kl_div = torch.distributions.kl.kl_divergence(raw_outputs_dist, pretrained_prob_dist)
            # # print("---- sentence level KL divergence ----- ", kl_div)
            
            # kl_div_sents = kl_div.mean(-1)
            # kl_div_batch = kl_div_sents.mean().item()
            # # print("---- batch level KL divergence ----- ", kl_div_batch)
            # kl_divs.append(kl_div_batch)
            # TODO double check + sign and whether the kl term requires grad
            # print("raw outputs before computing structural loss ", raw_outputs.shape)
            # print(raw_outputs.transpose(1,2).shape)
            # print(captions.shape)
            loss_structural = criterion(raw_outputs.transpose(1,2), captions[:, 1:]) #+ kl_coeff * kl_div_batch
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

            # scheduler_s.step()
            # scheduler_l.step()
            # print("Current scheduler LR: ", scheduler_s.get_last_lr(), scheduler_l.get_last_lr())     
            # Get training statistics.
            # perplexity computation questionable
            stats = 'Epoch [%d/%d], Step [%d/%d], Speaker loss: %.4f, Listener loss: %.4f, Perplexity: %5.4f, Accuracy: %.4f' % (epoch, num_epochs, i_step, total_steps, speaker_loss.item(), listener_loss.item(), torch.exp(loss_structural), accuracy.item())
            
            speaker_losses_structural.append(loss_structural.item())
            speaker_losses_functional.append(rl_grads.item())
            speaker_losses.append(speaker_loss.item())
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
            if i_step % 500 == 0:
                print('\r' + stats)
                # TODO double check
                # also compute the drift metrics during training to check the dynamics
                speaker_decoder.eval()
                init_hidden = speaker_decoder.init_hidden(1)
                for i in range(captions_pred.shape[0]): # iterate over sentences in batch
                    # semantic drift under pretrained captioning model
                    # decode caption to natural language for that
                    # outputs_ind = softmax(captions_pred[i])
                    # max_probs, cat_samples = torch.max(outputs_ind, dim = -1)
                    
                    eval_steps.append(i_step)
                    # structural drift under a pretrained LM
                    # decode caption to natural language for that
                    nl_caption = [data_loader.dataset.vocab.idx2word[w.item()] for w in captions_pred[i]]
                    nl_caption = " ".join(nl_caption)
                    structural_drift = drift_meter.structural_drift(nl_caption)
                    structural_drifts.append(structural_drift)
                    # also compute this for ground truth caption, as a reference value
                    nl_true_caption = [data_loader.dataset.vocab.idx2word[w.item()] for w in captions[i]]
                    nl_true_caption = " ".join(nl_true_caption)
                    structural_drift_true = drift_meter.structural_drift(nl_true_caption)
                    structural_drifts_true.append(structural_drift_true)
                    
                    ############
                    # structural drift under a pretrained LM
                    
                    # TODO check masking tokens after end
                    semantic_drift, _ = drift_meter.semantic_drift(captions_pred[i], both_images[i])
                    semantic_drifts.append(semantic_drift)
                    # print("----- Semantic drift ---- ", semantic_drift)
                    # for comparison, compute semantic drift of ground truth caption
                    semantic_drift_true, _ = drift_meter.semantic_drift(captions[i], both_images[i])
                    semantic_drifts_true.append(semantic_drift_true)
                    # print("----- Semantic drift ground truth ---- ", semantic_drift_true)

                    # overlap based drift metrics
                    discrete_overlap = drift_meter.compute_discrete_overlap(captions_pred[i], captions[i], dist_captions[i])
                    # print("Discrete overlap score ", discrete_overlap)
                    discrete_overlaps.append(discrete_overlap)
                    # TODO get embeddings of the captions for discrete overlap computation
                    with torch.no_grad():
                        _ , prediction_embs = speaker_decoder.forward(both_images[i].unsqueeze(0), captions_pred[i].unsqueeze(0), init_hidden)#decoder.embed(cat_samples)
                        _, target_embs = speaker_decoder.forward(both_images[i].unsqueeze(0), captions[i].unsqueeze(0), init_hidden) #decoder.embed(target_captions[i])
                        _ , distractor_embs = speaker_decoder.forward(both_images[i].unsqueeze(0), dist_captions[i].unsqueeze(0), init_hidden)#decoder.embed(cat_samples)
                    cont_overlap = drift_meter.compute_cont_overlap(prediction_embs[0], target_embs[0], distractor_embs[0])
                    # print("Continuous overlap ", cont_overlap)
                    cont_overlaps.append(cont_overlap)

                    # also compute image similarities
                    img_similarity_val = drift_meter.image_similarity(target_features[i], distractor_features[i])
                    image_similarities_val.append(img_similarity_val)
                    epochs_out.append(epoch)
                    ############

        # Save the weights.
        if epoch % save_every == 0:
            torch.save(speaker_decoder.state_dict(), os.path.join('./models', 'speaker-decoder-vocab49-metrics-full-3dshapes_ls02_fixed-%d.pkl' % epoch))
            # torch.save(speaker_encoder.state_dict(), os.path.join('./models', 'speaker-encoder-singleImgs-token0-vocab6000-%d.pkl' % epoch))
            torch.save(listener_rnn.state_dict(), os.path.join('./models', 'listener-rnn-vocab49-metrics-full-3dshapes_ls02_fixed-%d.pkl' % epoch))
            torch.save(listener_encoder.state_dict(), os.path.join('./models', 'listener-encoder-vocab49-metrics-full-3dshapes_ls02_fixed-%d.pkl' % epoch))
            
        # save the training metrics
        df_out = pd.DataFrame({
            "steps": steps,
            "speaker_s": speaker_losses_structural,
            "speaker_f": speaker_losses_functional,
            "speaker_loss": speaker_losses,
            "listener": listener_losses,
            "perplexities": perplexities,
            "accuracies": accuracies,
            "image_similarities": image_similarities,
            # "KL_divs": kl_divs,
        })
        df_out.to_csv(csv_out + "epoch_" + str(epoch) + ".csv", index=False )
        metrics_out = pd.DataFrame({
            "steps": eval_steps,
            "structural_drift_pred": structural_drifts,
            "structural_drift_true": structural_drifts_true,
            "semantic_drifts_true": semantic_drifts_true,
            "semantic_drifts_pred": semantic_drifts,      
            "discrete_overlaps": discrete_overlaps,
            "continuous_overlaps": cont_overlaps,
            "image_similarities": image_similarities_val,
            "epochs_out": epochs_out,

        })
        metrics_out.to_csv(csv_metrics + "epoch_" + str(epoch) + ".csv", index=False)
    # Close the training log file.
    f.close()
    pass