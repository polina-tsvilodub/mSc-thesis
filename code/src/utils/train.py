import sys
import os
import math
import torch
import numpy as np
import pandas as pd
from . import early_stopping
import torch.nn as nn 
from drift_metrics import metrics

def validate_model(
    data_loader_val,
    encoder,
    decoder,
):
    """
    Utility function for computing the validation performance of the speaker model while pre-training.
    """
    # init params
    val_running_loss = 0.0
    counter = 0
    total = 0
    embedded_imgs = torch.load("./notebooks/resnet_pretrain_embedded_val_imgs.pt")
    total_steps = math.ceil(len(data_loader_val.dataset.caption_lengths) / data_loader_val.batch_sampler.batch_size)
    print("Val total steps: ", total_steps+1)
    criterion = nn.CrossEntropyLoss().cuda() if torch.cuda.is_available() else nn.CrossEntropyLoss()

    encoder.eval()
    decoder.eval()

    # evaluate model on the batch
    with torch.no_grad():
        print("Validating the model...")
        for i in range(1, total_steps+1): 
             # get val indices
            indices = data_loader_val.dataset.get_func_train_indices()
            # create separate lists for retrieving the image emebddings
            # target_inds = torch.tensor([x[0] for x in indices]).long()
            # distractor_inds = torch.tensor([x[1] for x in indices]).long()

            new_sampler = torch.utils.data.sampler.SubsetRandomSampler(indices=indices)
            data_loader_val.batch_sampler.sampler = new_sampler

            counter += 1
            # print("Counter: ", counter)
            # Obtain the batch.
            targets, distractors, target_captions = next(iter(data_loader_val))

            # Move batch of images and captions to GPU if CUDA is available.
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            targets = targets.to(device)
            distractors = distractors.to(device)
            target_captions = target_captions.to(device)

            # Pass the inputs through the CNN-RNN model.
            # after retrieving the resnet embeddings
            # target_features = torch.index_select(embedded_imgs, 0, target_inds)
            # distractor_features = torch.index_select(embedded_imgs, 0, distractor_inds)

            # compute val predictions and loss
            target_features = encoder(targets)#encoder(target_features)
            distractor_features = encoder(distractors) #encoder(distractor_features)

            both_images = torch.cat((target_features, distractor_features), dim=-1)
            outputs = decoder(both_images, target_captions)
            
            # The size of the vocabulary.
            vocab_size = len(data_loader_val.dataset.vocab)

            # Calculate the batch loss.
            loss = criterion(outputs.contiguous().view(-1, vocab_size), target_captions[:, 1:].reshape(-1))  

            val_running_loss += loss.item()
            # print("val running loss: ", val_running_loss)
            
        
        val_loss = val_running_loss / counter
        
        return val_loss


def pretrain_speaker(
    log_file,
    num_epochs,
    total_steps,
    data_loader,
    data_loader_val, 
    # encoder,
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
        encoder: CNN
            Pretrained ResNet-50 instance.
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
    csv_out = "../../data/metrics-test-pretrain_3dshapes_512dim_losses_47vocab_rs1234_wEmb_cont_"
    val_csv_out = "../../data/pretrain_3dshapes_512dim_val_losses_47vocab_cont"
    csv_metrics = "metrics-test-pretrain_3dshapes_512dim_metrics_47vocab_wEmb_"
    
    speaker_losses=[]
    perplexities = []
    steps = []
    val_losses, val_steps = [], []
    eval_steps = []
    structural_drifts_true = []
    structural_drifts = []
    # init the early stopper
    early_stopper = early_stopping.EarlyStopping(patience=3, min_delta=0.03)

    # embedded_imgs = torch.load("3dshapes_all_ResNet_features_reshaped_all.pt") #torch.cat(( torch.load("3dshapes_all_ResNet_features_reshaped_23000_first.pt"), torch.load("3dshapes_all_ResNet_features_reshaped_240000_first.pt") ), dim = 0) #torch.load("COCO_train_ResNet_features_reshaped.pt")
    # init drift meter for tracking structural drift for reference
    drift_meter = metrics.DriftMeter(
        # semantic_encoder="models/encoder-earlystoppiing-4_semantic-drift.pkl", 
        semantic_decoder="models/decoder-3dshapes-512dim-4000vocab-3.pkl", 
        structural_model="transfo-xl-wt103",  
        embed_size=512, 
        vis_embed_size=512, 
        hidden_size=512,
        # TODO this will have to be retrained
        vocab=len(data_loader.dataset.vocab)
    )
    softmax = nn.Softmax(dim=-1)
    # hidden = decoder.init_hidden(64)
    for epoch in range(1, num_epochs+1):
        # encoder.train()
        decoder.train()

        # decoder.init_hidden(batch_size=64)
        for i_step in range(1, total_steps+1):
            # set models into training mode
            hidden = decoder.init_hidden(batch_size=64)
                    
            # Randomly sample a caption length, and sample indices with that length.
            indices = data_loader.dataset.get_func_train_indices()
            # indices = [(1,0)]
            # create separate lists for retrieving the image emebddings
            # target_inds = torch.tensor([x[0] for x in indices]).long()
            # distractor_inds = torch.tensor([x[1] for x in indices]).long()
            # print("Indices targets: ", target_inds)
            # print("Indices distractors: ", distractor_inds)
            # print("Indices: ", indices)
            # Create and assign a batch sampler to retrieve a batch with the sampled indices.
            new_sampler = torch.utils.data.sampler.SubsetRandomSampler(indices=indices)
            data_loader.batch_sampler.sampler = new_sampler
            
            # Obtain the batch.
            targets, distractors, target_features, distractor_features, target_captions = next(iter(data_loader)) # distractor_captions
            
            # Move batch of images and captions to GPU if CUDA is available.
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            targets = targets.to(device)
            distractors = distractors.to(device)
            target_captions = target_captions.to(device)
            # print("TARGET CAPTIONS shape: ", target_captions.shape)

            # Zero the gradients (reset).
            decoder.zero_grad()
            # encoder.zero_grad()
            
            # Pass the inputs through the CNN-RNN model.
            # after retrieving the resnet embeddings
            # target_features = torch.index_select(embedded_imgs, 0, target_inds)
            # distractor_features = torch.index_select(embedded_imgs, 0, distractor_inds)

            # target_features = encoder(targets)#encoder(target_features)
            # print("Script encoder output: ", target_features)
            # distractor_features = encoder(distractors) #encoder(distractor_features)
            # print("embedded resnet features shape: ", target_features.shape)
            # concat image features
            both_images = torch.cat((target_features.unsqueeze(1), distractor_features.unsqueeze(1)), dim=1) # [target_features, distractor_features]
            outputs, hidden = decoder(both_images, target_captions, hidden)
            
            # The size of the vocabulary.
            vocab_size = len(data_loader.dataset.vocab)

            # Calculate the batch loss.
            # print("Outputs shape for loss: ", outputs.shape)
            # print("transposed output: ", outputs.transpose(1,2).shape)
            # print("Targets shape for loss: ", target_captions.shape)
            loss = criterion(outputs.transpose(1,2), target_captions[:, 1:]) # 
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
            # print(loss.item())
            sys.stdout.flush()
            
            # Print training statistics to file.
            f.write(stats + '\n')
            f.flush()
            
            # Print training statistics (on different line).
            if i_step % 1 == 0:
                print('\r' + stats)
                # also compute structural drift, for potential comparison to reference game setting
                for i in range(outputs.shape[0]): # iterate over sentences in batch
                    # structural drift under a pretrained LM
                    # decode caption to natural language for that
                    eval_steps.append(i_step)
                    outputs_ind = softmax(outputs[i])
                    max_probs, cat_samples = torch.max(outputs_ind, dim = -1)

                    # need to be transformed argmaxxed indices
                    # TODO check masking tokens after end
                    semantic_drift = drift_meter.semantic_drift(cat_samples, both_images[i])
                    print("----- Semantic drift ---- ", semantic_drift)
                    

                    nl_caption = [data_loader.dataset.vocab.idx2word[w.item()] for w in cat_samples]
                    nl_caption = " ".join(nl_caption)
                    structural_drift = drift_meter.structural_drift(nl_caption)
                    structural_drifts.append(structural_drift.item())
                    # also compute this for ground truth caption, as a reference value
                    # caption_ind = softmax(targets_captions[i])
                    nl_true_caption = [data_loader.dataset.vocab.idx2word[w.item()] for w in target_captions[i]]
                    nl_true_caption = " ".join(nl_true_caption)
                    structural_drift_true = drift_meter.structural_drift(nl_true_caption)
                    structural_drifts_true.append(structural_drift_true.item())

                    # TODO overlap based drift metrics

        # Save the weights.
        if epoch % save_every == 0:
            torch.save(decoder.state_dict(), os.path.join('./models', 'metrics-test-decoder-3dshapes-512dim-47vocab-rs1234-wEmb-cont-%d.pkl' % epoch))
            # torch.save(encoder.state_dict(), os.path.join('./models', 'encoder-noEnc-prepend-1024dim-4000vocab-wResNet-%d.pkl' % epoch))

            # compute validation loss
            # with torch.no_grad():
                # val_loss = validate_model(
                #     data_loader_val,
                #     encoder,
                #     decoder,
                # )     
                # # torch.save(decoder.state_dict(), os.path.join('./models', 'decoder-2imgs-earlystopping-%d-step-%d.pkl' % (epoch, i_step)))
                # # torch.save(encoder.state_dict(), os.path.join('./models', 'encoder-2imgs-earlystoppiing-%d-step-%d.pkl' % (epoch, i_step)))
                
                # val_stats = 'Epoch [%d/%d], Step [%d/%d], Validation loss: %.4f' % (epoch, num_epochs, i_step, total_steps, val_loss)
                # print(val_stats)
                # f.write(val_stats + '\n')
                # f.flush()   
                
                # val_losses.append(val_loss)
                # val_steps.append(i_step)
                # # update stooper
                # early_stopper(val_loss)
                # # check if we want to break
                # if early_stopper.early_stop:
                #     # save the models
                #     torch.save(decoder.state_dict(), os.path.join('./models', 'decoder-byImage-prepend-1024dim-earlystopping-6000vocab-wResNet-%d.pkl' % epoch))
                #     torch.save(encoder.state_dict(), os.path.join('./models', 'encoder-byImage-prepend-1024dim-earlystopping-6000vocab-wResNet-%d.pkl' % epoch))
                #     print("Early stopping steps loop!")
                #     break

        # save the training metrics
        df_out = pd.DataFrame({
            "steps": steps,
            "losses": speaker_losses,
            "perplexities": perplexities
        })
        df_out.to_csv(csv_out + "epoch_" + str(epoch) + ".csv", index=False )

        df_val_out = pd.DataFrame({
            "steps": val_steps,
            "val_losses": val_losses
        })
        # df_val_out.to_csv(val_csv_out + "epoch_" + str(epoch) + ".csv", index=False)

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