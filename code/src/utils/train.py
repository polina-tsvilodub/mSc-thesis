import sys
import os
import math
import torch
import numpy as np
import pandas as pd
from . import early_stopping
import torch.nn as nn 

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
            target_inds = torch.tensor([x[0] for x in indices]).long()
            distractor_inds = torch.tensor([x[1] for x in indices]).long()

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
            target_features = torch.index_select(embedded_imgs, 0, target_inds)
            distractor_features = torch.index_select(embedded_imgs, 0, distractor_inds)
            # concat image features
            both_images = torch.cat((target_features, distractor_features), dim=-1)

            # compute val predictions and loss
            both_images_features = encoder(both_images)
            # distractor_features = encoder(distractor_features)

            # both_images = torch.cat((target_features, distractor_features), dim=-1)
            outputs = decoder(both_images_features, target_captions)
            
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
    encoder,
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
    csv_out = "../../data/pretraining_2imgs_token0_1024dim_losses_2000vocab_"
    val_csv_out = "../../data/pretraining_2imgs_token0_1024dim_val_losses_2000vocab_"

    speaker_losses=[]
    perplexities = []
    steps = []
    val_losses, val_steps = [], []
    # init the early stopper
    early_stopper = early_stopping.EarlyStopping(patience=3, min_delta=0.03)

    embedded_imgs = torch.load("./notebooks/resnet_pretrain_embedded_imgs.pt")

    for epoch in range(1, num_epochs+1):
        encoder.train()
        decoder.train()

        decoder.init_hidden(batch_size=64)
        for i_step in range(1, total_steps+1):
            # set models into training mode
            
                    
            # Randomly sample a caption length, and sample indices with that length.
            indices = data_loader.dataset.get_func_train_indices()
            # create separate lists for retrieving the image emebddings
            target_inds = torch.tensor([x[0] for x in indices]).long()
            distractor_inds = torch.tensor([x[1] for x in indices]).long()
            # Create and assign a batch sampler to retrieve a batch with the sampled indices.
            new_sampler = torch.utils.data.sampler.SubsetRandomSampler(indices=indices)
            data_loader.batch_sampler.sampler = new_sampler
            
            # Obtain the batch.
            targets, distractors, target_captions = next(iter(data_loader))

            # Move batch of images and captions to GPU if CUDA is available.
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            targets = targets.to(device)
            distractors = distractors.to(device)
            target_captions = target_captions.to(device)
            
            # Zero the gradients (reset).
            decoder.zero_grad()
            encoder.zero_grad()
            
            # Pass the inputs through the CNN-RNN model.
            # after retrieving the resnet embeddings
            target_features = torch.index_select(embedded_imgs, 0, target_inds)
            distractor_features = torch.index_select(embedded_imgs, 0, distractor_inds)

            # concat image features
            both_images = torch.cat((target_features, distractor_features), dim=-1)

            both_images_features = encoder(both_images)
            # distractor_features = encoder(distractor_features)

            outputs = decoder(both_images_features, target_captions)
            
            # The size of the vocabulary.
            vocab_size = len(data_loader.dataset.vocab)

            # Calculate the batch loss.
            loss = criterion(outputs.contiguous().view(-1, vocab_size), target_captions[:, 1:].reshape(-1))
            
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
            if i_step % print_every == 0:
                print('\r' + stats)

        # Save the weights.
        if epoch % save_every == 0:
            torch.save(decoder.state_dict(), os.path.join('./models', 'decoder-2imgs-token0-1024dim-2000vocab-%d.pkl' % epoch))
            torch.save(encoder.state_dict(), os.path.join('./models', 'encoder-2imgs-token0-1024dim-2000vocab-%d.pkl' % epoch))

            # compute validation loss
            with torch.no_grad():
                val_loss = validate_model(
                    data_loader_val,
                    encoder,
                    decoder,
                )     
                # torch.save(decoder.state_dict(), os.path.join('./models', 'decoder-2imgs-earlystopping-%d-step-%d.pkl' % (epoch, i_step)))
                # torch.save(encoder.state_dict(), os.path.join('./models', 'encoder-2imgs-earlystoppiing-%d-step-%d.pkl' % (epoch, i_step)))
                
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
                    torch.save(decoder.state_dict(), os.path.join('./models', 'decoder-2imgs-token0-1024dim-earlystopping-2000vocab-%d.pkl' % epoch))
                    torch.save(encoder.state_dict(), os.path.join('./models', 'encoder-2imgs-token0-1024dim-earlystoppiing-2000vocab-%d.pkl' % epoch))
                    print("Early stopping steps loop!")
                    break

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
        df_val_out.to_csv(val_csv_out + "epoch_" + str(epoch) + ".csv", index=False)

        # also break the epochs 
        if early_stopper.early_stop:
            print("Early stopping epochs loop!")
            break

    # Close the training log file.
    f.close()