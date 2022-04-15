import sys
import torch
import numpy as np

def pretrain_speaker(
    log_file,
    num_epochs,
    total_steps,
    data_loader, 
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

    for epoch in range(1, num_epochs+1):
        
        for i_step in range(1, total_steps+1):
                    
            # Randomly sample a caption length, and sample indices with that length.
            indices = data_loader.dataset.get_train_indices()
            # Create and assign a batch sampler to retrieve a batch with the sampled indices.
            new_sampler = torch.utils.data.sampler.SubsetRandomSampler(indices=indices)
            data_loader.batch_sampler.sampler = new_sampler
            
            # Obtain the batch.
            images, captions = next(iter(data_loader))

            # Move batch of images and captions to GPU if CUDA is available.
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            images = images.to(device)
            captions = captions.to(device)
            
            # Zero the gradients (reset).
            decoder.zero_grad()
            encoder.zero_grad()
            
            # Pass the inputs through the CNN-RNN model.
            features = encoder(images)
            outputs = decoder(features, captions)
            
            # The size of the vocabulary.
            vocab_size = len(data_loader.dataset.vocab)

            # Calculate the batch loss.
            loss = criterion(outputs.contiguous().view(-1, vocab_size), captions.reshape(-1))
            
            # Backward pass.
            loss.backward()
            
            # Update the parameters in the optimizer.
            optimizer.step()
                
            # Get training statistics.
            stats = 'Epoch [%d/%d], Step [%d/%d], Loss: %.4f, Perplexity: %5.4f' % (epoch, num_epochs, i_step, total_steps, loss.item(), np.exp(loss.item()))
            
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
            torch.save(decoder.state_dict(), os.path.join('./models', 'decoder-%d.pkl' % epoch))
            torch.save(encoder.state_dict(), os.path.join('./models', 'encoder-%d.pkl' % epoch))

    # Close the training log file.
    f.close()