import sys
import os
import math
import torch
import numpy as np
import pandas as pd
import torch.nn as nn 
from random import shuffle

from drift_metrics import metrics
from . import update_policy

def validate_game_compute_metrics(
    speaker_decoder,
    drift_meter,
    data_loader_val,
    decoding_strategy,
    epoch,
    pairs,
):
    """
    Helper for validating the speaker and computing all drift metrics while training
    the agents in a reference game.

    Arguments:
    ---------

    Returns:
    -------

    """
    # TODO track at which main loop steps this is happening
    val_running_loss = 0.0
    val_running_ppl = 0.0
    counter = 0

    eval_steps = []
    structural_drifts_true = []
    structural_drifts = []
    semantic_drifts = []
    semantic_drifts_true = []
    discrete_overlaps = []
    cont_overlaps = []
    epochs_out = []
    val_ppl = []
    val_losses = []
    image_similarities_val = []

    speaker_decoder.eval()

    total_steps = math.floor(len(data_loader_val.dataset.ids) / data_loader_val.batch_sampler.batch_size)

    init_hidden = speaker_decoder.init_hidden(data_loader_val.batch_sampler.batch_size)

    criterion = nn.CrossEntropyLoss().cuda() if torch.cuda.is_available() else nn.CrossEntropyLoss()

    for i in range(1, total_steps+1):
        # compute validation loss and PPL
        with torch.no_grad():
            if pairs == "random":
                indices = data_loader_val.dataset.get_func_train_indices(i)
            else:
                indices = data_loader_val.dataset.get_func_similar_train_indices()

            new_sampler = torch.utils.data.sampler.SubsetRandomSampler(indices=indices)
            data_loader_val.batch_sampler.sampler = new_sampler

            counter += 1
            # obtain batch
            images1, images2, target_features, distractor_features, captions, dist_captions = next(iter(data_loader_val))
            # do forward pass
            both_images = torch.cat((target_features.unsqueeze(1), distractor_features.unsqueeze(1)), dim=1)
            # sample caption from speaker 
            # get predicted caption and its log probability
            captions_pred, log_probs, raw_outputs, entropies = speaker_decoder.sample(both_images, max_sequence_length=captions.shape[1]-1, decoding_strategy=decoding_strategy)
            # compute val loss and PPL
            loss_structural = criterion(raw_outputs.transpose(1,2), captions[:, 1:]) #+ kl_coeff * kl_div_batch
            val_losses.append(loss_structural.item())
            ppl = torch.exp(loss_structural).item()
            val_running_ppl += ppl
            val_ppl.append(ppl)
            val_running_loss += loss_structural.item()        

            eval_steps.append(i)
            # structural drift under a pretrained LM
            # decode caption to natural language for that
            nl_captions = [[data_loader_val.dataset.vocab.idx2word[w.item()] for w in s] for s in captions_pred]
            nl_captions = [" ".join(nl_caption) for nl_caption in nl_captions]
            
            structural_drift = drift_meter.structural_drift(nl_captions)
            
            structural_drifts.append(structural_drift.item())
            # also compute this for ground truth caption, as a reference value
            nl_true_captions = [[data_loader_val.dataset.vocab.idx2word[w.item()] for w in s] for s in captions]
            nl_true_captions = [" ".join(nl_true_caption) for nl_true_caption in nl_true_captions]
            structural_drift_true = drift_meter.structural_drift(nl_true_captions)
            structural_drifts_true.append(structural_drift_true.item())
            
            ############
            # structural drift under a pretrained LM
            
            # TODO check masking tokens after end
            semantic_drift, _ = drift_meter.semantic_drift(captions_pred, both_images)
            semantic_drifts.append(semantic_drift)
            # print("----- Semantic drift ---- ", semantic_drift)
            # for comparison, compute semantic drift of ground truth caption
            semantic_drift_true, _ = drift_meter.semantic_drift(captions, both_images)
            semantic_drifts_true.append(semantic_drift_true)
            # print("----- Semantic drift ground truth ---- ", semantic_drift_true)

            # overlap based drift metrics
            discrete_overlap = drift_meter.compute_discrete_overlap(captions_pred, captions, dist_captions)
            discrete_overlaps.append(discrete_overlap)
            
            # get embeddings of the captions for discrete overlap computation
            _ , prediction_embs = speaker_decoder.forward(both_images, captions_pred, init_hidden)#decoder.embed(cat_samples)
            _, target_embs = speaker_decoder.forward(both_images, captions, init_hidden) #decoder.embed(target_captions[i])
            _ , distractor_embs = speaker_decoder.forward(both_images, dist_captions, init_hidden)#decoder.embed(cat_samples)
            cont_overlap = drift_meter.compute_cont_overlap(prediction_embs[0].squeeze(), target_embs[0].squeeze(), distractor_embs[0].squeeze())
            cont_overlaps.append(cont_overlap)

            # also compute image similarities
            img_similarity_val = drift_meter.image_similarity(target_features, distractor_features)
            image_similarities_val.append(img_similarity_val)
            epochs_out.append(epoch)
    print("out shape of str drift list: ", len(structural_drifts), structural_drifts[0])
    print("out shape of sem drift list: ", len(semantic_drifts), semantic_drifts[0])
    print("out shape of img sim list: ", len(image_similarities_val), image_similarities_val[0])
    print("out shape of cont ovelap list: ", len(cont_overlaps), cont_overlaps[0])
    
    
    val_loss_out = val_running_loss / counter
    val_ppl_out = val_running_ppl / counter
    print("Final val loss ", val_loss_out)
    print("Final val ppl ", val_ppl_out)
    print("eval steps: ", eval_steps)
    print("epochs out ", epochs_out)
    # TODO check if should write a file here or return means over the different batches            
    # possibly return lists and extend the extant ones in the main loop
    # check if I need a particular type of return for potential early stopping
    return val_loss_out, val_ppl_out, val_losses, val_ppl, epochs_out, eval_steps,\
        structural_drifts, structural_drifts_true, semantic_drifts, semantic_drifts_true,\
        cont_overlaps, discrete_overlaps, image_similarities_val

def play_game(
    log_file,
    num_epochs,
    total_steps,
    data_loader,
    data_loader_val, 
    speaker_decoder,
    listener_encoder, 
    listener_rnn,
    criterion,
    weights_path,
    print_every,
    save_every,
    train_losses_file,
    train_metrics_file,
    val_losses_file,
    val_metrics_file,
    experiment,
    lambda_s,
    pretrained_decoder_file,
    decoding_strategy,
    mean_baseline,
    entropy_weight,
    pairs="random",
    use_encode_ls=False,
    use_tf_ls=False,
):
    # Open the training log file.
    f = open(log_file, 'w')

    csv_out = train_losses_file + str(pretrained_decoder_file).replace("models/", "").replace(".pkl", "") #"functional_training_losses_COCO_lf01_"
    csv_metrics = train_metrics_file + str(pretrained_decoder_file).replace("models/", "").replace(".pkl", "") #"functional_training_COCO_language_drift_metrics_lf01_"
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
    mean_baselines = []

    eval_steps = []
    structural_drifts_true = []
    structural_drifts = []
    semantic_drifts = []
    semantic_drifts_true = []
    discrete_overlaps = []
    cont_overlaps = []
    epochs_out = []
    val_loss_avg = []
    val_ppl_avg = []
    val_losses_all = []
    val_ppl_all = []
    sanity_check_inds = []

    lambda_s = lambda_s
    lambda_f = 1 - lambda_s
    kl_coeff = 0.1
    # torch.autograd.set_detect_anomaly(True)

    # embedded_imgs = torch.load("COCO_train_ResNet_features_reshaped.pt")
    speaker_params = list(speaker_decoder.embed.parameters()) + list(speaker_decoder.lstm.parameters()) + list(speaker_decoder.linear.parameters()) + list(speaker_decoder.project.parameters()) 
    listener_params = list(listener_rnn.lstm.parameters()) + list(listener_encoder.embed.parameters()) 
    
    # Define the optimizer.
    speaker_optimizer = torch.optim.Adam(speaker_params, lr=0.001, betas=(0.9, 0.999), eps=1e-08)
    listener_optimizer = torch.optim.Adam(listener_params, lr=0.001, betas=(0.9, 0.999), eps=1e-08)

    # LR scheduler
    # scheduler_s = torch.optim.lr_scheduler.StepLR(speaker_optimizer, step_size=1000, gamma=0.5)
    # scheduler_l = torch.optim.lr_scheduler.StepLR(listener_optimizer, step_size=1000, gamma=0.5)

    # init the drift metrics class
    drift_meter = metrics.DriftMeter(
        # semantic_decoder="models/decoder-3dshapes-512dim-47vocab-rs1234-wEmb-short-5.pkl",
        # semantic_decoder="models/decoder-3dshapes-512dim-49vocab-rs1234-exh-3.pkl", 
        semantic_decoder=pretrained_decoder_file,#"models/decoder-noEnc-prepend-512dim-4000vocab-rs1234-wEmb-cont-7.pkl", 
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

    # create a list of "steps" which can be shuffled as a proxy for shuffling the indices of the images used for batching
    steps_nums = list(range(1, total_steps+1))
    
    with torch.autograd.set_detect_anomaly(True):
        for epoch in range(1, num_epochs+1):
            # manually shuffle the indices
            shuffle(steps_nums)
            for i, i_step in enumerate(steps_nums):
                hidden = speaker_decoder.init_hidden(batch_size=data_loader.batch_sampler.batch_size)
                # set mode of the models
                speaker_decoder.train()
                listener_encoder.train()
                listener_rnn.train()

                # Randomly sample a caption length, and sample indices with that length.
                if pairs == "random":
                    indices_pairs = data_loader.dataset.get_func_train_indices(i_step)
                else:
                    indices_pairs = data_loader.dataset.get_func_similar_train_indices()

                sanity_check_inds.append(indices_pairs)
                # Create and assign a batch sampler to retrieve a target batch with the sampled indices.
                new_sampler_pairs = torch.utils.data.sampler.SubsetRandomSampler(indices=indices_pairs)
                
                data_loader.batch_sampler.sampler = new_sampler_pairs
                # Obtain the target batch.
                images1, images2, target_features, distractor_features, captions, dist_captions = next(iter(data_loader))
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
                # addition of TF computation in decoding a decoding based expt is going to mean that we have to 
                # do this manually step weise like in the pretraining
                if use_encode_ls:
                    raw_outputs, _ = speaker_decoder(both_images, captions, hidden)
                    norm_outputs = softmax(raw_outputs)
                    captions_probs, captions_pred = torch.max(norm_outputs, dim = -1)
                    log_probs = torch.log(captions_probs)
                    entropies = -log_probs * captions_probs
                    
                    # end_mask = captions_pred.size(-1) - (torch.eq(captions_pred, 1).to(torch.int64).cumsum(dim=-1) > 0).sum(dim=-1)
                    # # include the END token
                    # end_inds = end_mask.add_(1).clamp_(max=captions_pred.size(-1)) # shape: (batch_size,)
                    # for pos, i in enumerate(end_inds):  
                    #     # print("end inds ", i)
                    #     # zero out log Ps and entropies
                    #     log_probs[pos, i:] = 0
                    #     entropies[pos, i:] = 0
                else:                
                # sample caption from speaker 
                # get predicted caption and its log probability
                    captions_pred, log_probs, raw_outputs, entropies = speaker_decoder.sample(both_images, max_sequence_length=captions.shape[1]-1, decoding_strategy=decoding_strategy)
                print("---- GROUND TRUTH CAPTIONS ---- ", update_policy.clean_sentence(captions, data_loader)[:2])
                print("---- PREDICTED CAPTIONS ---- ", update_policy.clean_sentence(captions_pred, data_loader)[:2])
                
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
                hiddens_scores, hidden = listener_rnn(captions_pred)
                features = torch.cat((features1.unsqueeze(1), features2.unsqueeze(1)), dim=1)
                predictions, scores = listener_encoder(features, hidden.squeeze(0)) 
            
                ######
                # RL step
                # if target index and output index match, 1, else -1
                accuracy = torch.sum(torch.eq(targets_list, predictions).to(torch.int64))/predictions.shape[0]
                accuracies.append(accuracy.item())
                rewards = [1 if x else -1 for x in torch.eq(targets_list, predictions).tolist()]
                #####
                # mean baseline
                if mean_baseline:
                    b = mean_baseline.get()
                    mean_baseline.update(rewards)
                    rewards = torch.tensor(rewards)
                    rewards = rewards - b 
                    try:
                        mean_baselines.append(b.mean().item())
                    except AttributeError:
                        mean_baselines.append(b)
                ####
                # compute REINFORCE update
                rl_grads = update_policy.update_policy(rewards, log_probs, entropies, entropy_weight=entropy_weight) # check if log probs need to be stacked first
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
                # KL divergence for regularization
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
                loss_structural = criterion(raw_outputs.transpose(1,2), captions[:, 1:]) #+ kl_coeff * kl_div_batch
                speaker_loss =  lambda_s*loss_structural + lambda_f*rl_grads
                
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
                # Get training statistics.
                stats = 'Epoch [%d/%d], Step [%d/%d], Speaker loss: %.4f, Listener loss: %.4f, Perplexity: %5.4f, Accuracy: %.4f' % (epoch, num_epochs, i, total_steps, speaker_loss.item(), listener_loss.item(), torch.exp(loss_structural), accuracy.item())
                
                speaker_losses_structural.append(loss_structural.item())
                speaker_losses_functional.append(rl_grads.item())
                speaker_losses.append(speaker_loss.item())
                listener_losses.append(listener_loss.item())
                perplexities.append(torch.exp(loss_structural).item())
                steps.append(i)
                
                # Print training statistics (on same line).
                print('\r' + stats, end="")
                sys.stdout.flush()
                
                # Print training statistics to file.
                f.write(stats + '\n')
                f.flush()
                
                # Print training statistics (on different line).
                if i % 500 == 0:
                    print('\r' + stats)
                    # TODO double check
                    # also compute the drift metrics during training to check the dynamics

                    val_loss_out, val_ppl_out, val_losses, val_ppl, epoch_out, eval_step,\
                        structural_drift, structural_drift_true, semantic_drift, semantic_drift_true,\
                        cont_overlap, discrete_overlap, image_similarity_val = validate_game_compute_metrics(
                            speaker_decoder=speaker_decoder,
                            drift_meter=drift_meter,
                            data_loader_val=data_loader_val,
                            decoding_strategy=decoding_strategy,
                            epoch=epoch,
                            pairs=pairs,
                        )
                    val_loss_avg.append(val_loss_out)
                    val_ppl_avg.append(val_ppl_out)
                    val_losses_all.extend(val_losses)
                    val_ppl_all.extend(val_ppl)
                    epochs_out.extend(epoch_out)
                    eval_steps.extend(eval_step)
                    structural_drifts.extend(structural_drift)
                    structural_drifts_true.extend(structural_drift_true)
                    semantic_drifts.extend(semantic_drift)
                    semantic_drifts_true.extend(semantic_drift_true)
                    cont_overlaps.extend(cont_overlap)
                    discrete_overlaps.extend(discrete_overlap)
                    image_similarities_val.extend(image_similarity_val)

            # Save the weights.
            if epoch % save_every == 0:
                speaker_trained_file = os.path.join('./models', 'speaker_greedyDecoding_descTF_paddedLoader_pretrained_negH' + experiment + '_vocab4000_' + str(lambda_s) + 'ls_' + decoding_strategy + "_decoding_fixed_" + str(pretrained_decoder_file).replace("models/", "").replace(".pkl", "") )
                torch.save(speaker_decoder.state_dict(), speaker_trained_file + '%d.pkl' % epoch)
                listener_rnn_trained_file = os.path.join('./models', 'listener_rnn_greedyDecoding_descTF_paddedLoader_pretrained_negH' + experiment + '_vocab4000_' + 'ls_' + str(lambda_s) + decoding_strategy + "_decoding_fixed_"+ str(pretrained_decoder_file).replace("models/", "").replace(".pkl", "") )
                listener_encoder_trained_file = os.path.join('./models', 'listener_encoder_greedyDecoding_descTF_paddedLoader_pretrained_negH' + experiment + '_vocab4000_' + 'ls_' + str(lambda_s) + decoding_strategy + "_decoding_fixed_"+ str(pretrained_decoder_file).replace("models/", "").replace(".pkl", "") )
                torch.save(listener_rnn.state_dict(), listener_rnn_trained_file +'%d.pkl' % epoch)
                torch.save(listener_encoder.state_dict(), listener_encoder_trained_file+'%d.pkl' % epoch )

                torch.save(sanity_check_inds, "sanity_check_indices_refgame.pt")
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
                "mean_mean_baselines": mean_baselines,
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
                "losses": val_losses_all,
                "ppl": val_ppl_all,

            })
            metrics_out.to_csv(csv_metrics + "epoch_" + str(epoch) + ".csv", index=False)
    # Close the training log file.
    f.close()

    return speaker_losses, speaker_losses_structural,\
        speaker_losses_functional, perplexities, listener_losses,\
        accuracies, structural_drifts, structural_drifts_true, semantic_drifts,\
        semantic_drifts_true, discrete_overlaps, cont_overlaps, image_similarities_val,\
        epochs_out, val_losses_all, val_ppl_all, val_loss_avg, val_ppl_avg, eval_steps
