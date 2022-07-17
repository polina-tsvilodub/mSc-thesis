# import tensorflow as tf
import os
from utils.build_dataset import make_dataset, get_loader, get_loader_3dshapes
from utils.download import maybe_download_and_extract
from utils.train import pretrain_speaker
from utils.early_stopping import EarlyStopping
import torch
from torchvision import transforms
import torch.nn as nn
from agents.speaker import DecoderRNN
from agents.resnet_encoder import EncoderCNN, EncoderMLP
import math
import random
import argparse

# set random seed
torch.manual_seed(1234)
random.seed(1234)

#######
# HYPERPARAMS 
########

def run_pretraining(
    SPEAKER_PRETRAINED,
    SPEAKER_WEIGHTS,
    EPOCHS,
    EXPERIMENT,
    DATASET,
    VAL_DATASET,
    MAX_SEQUENCE_LEN,
    BATCH_SIZE,
    LOG_FILE,
    VOCAB_FILE,
    TRAIN_LOSSES_FILE,
    TRAIN_METRICS_FILE,
    VAL_METRICS_FILE,
    NUM_IMG,
    WEIGHTS_PATH,
):
    """
    Utility wrapper for configuting and calling the pretraining.
    """
    # Desired image dimensions
    IMAGE_SIZE = 256

    # Vocabulary parameters
    VOCAB_THRESHOLD = 25 # minimum word count threshold
    VOCAB_FROM_FILE = True # if True, load existing vocab file
    VOCAB_FROM_PRETRAINED = False
    # Fixed length allowed for any sequence
    MAX_SEQUENCE_LENGTH = int(MAX_SEQUENCE_LEN)
    # path / name of vocab file
    VOCAB_FILE = VOCAB_FILE

    # Model Dimensions
    EMBED_SIZE = 512 # 1024 # dimensionality of word embeddings
    HIDDEN_SIZE = 512 # number of features in hidden state of the LSTM decoder
    VISUAL_EMBED_SIZE = 512 # dimensionality of visual embeddings

    # Other training parameters
    BATCH_SIZE = int(BATCH_SIZE)
    EPOCHS = int(EPOCHS) # number of training epochs
    PRINT_EVERY = 200 # window for printing average loss (steps)
    SAVE_EVERY = 1 # frequency of saving model weights (epochs)
    LOG_FILE = LOG_FILE # name of file with saved training loss and perplexity
    MODE= 'train' # network mode
    WEIGHTS_PATH=WEIGHTS_PATH
    NUM_VAL_IMGS=3700

    # data download params
    DOWNLOAD_DIR_TRAIN = "../../data/train"
    DOWNLOAD_DIR_VAL = "../../data/val"

    BASE_URL = "http://images.cocodataset.org/"
    domains_list = {
        DOWNLOAD_DIR_VAL: "zips/val2014.zip", 
        DOWNLOAD_DIR_TRAIN: ["annotations/annotations_trainval2014.zip",
        "zips/train2014.zip"], 
    }
    # path to pre-saved image features file
    embedded_imgs = torch.load("train_logs/COCO_train_ResNet_features_reshaped_dict.pt")

    #########

    print("Beginning speaker pretraining script...")

    if EXPERIMENT == "coco":

        # path to pre-saved image features file
        embedded_imgs = torch.load("train_logs/COCO_train_ResNet_features_reshaped_dict.pt")

        # download data 
        for filename in domains_list[DOWNLOAD_DIR_TRAIN]:
            url = BASE_URL + filename
            print("Downloading ", filename)
            maybe_download_and_extract(
                base_url = BASE_URL,
                filename = filename,
                download_dir = DOWNLOAD_DIR_TRAIN,
            )
            

        # image preprocessing
        # no cropping because relevant objects might get cropped and the grounding wouldn't be sensible anymore
        transform_train = transforms.Compose([ 
            transforms.Resize(IMAGE_SIZE),                   # resize image resolution to 256 (along smaller edge, the other proportionally)
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),               # horizontally flip image with probability=0.5
            transforms.ToTensor(),                           # convert the PIL Image to a tensor
            transforms.Normalize((0.485, 0.456, 0.406),      # normalize image for pre-trained model, tuples for means and std for the three img channels
                                (0.229, 0.224, 0.225))])

        # Build data loader, allowing to iterate over records from annotations file
        data_loader_train = get_loader(
            transform=transform_train,
            mode=MODE,
            batch_size=BATCH_SIZE,
            vocab_threshold=VOCAB_THRESHOLD,
            vocab_file=VOCAB_FILE,
            vocab_from_file=VOCAB_FROM_FILE,
            download_dir=DOWNLOAD_DIR_TRAIN,
            embedded_imgs=embedded_imgs,
        )

        data_loader_val = get_loader(
            transform=transform_train,
            mode="train",
            batch_size=BATCH_SIZE,
            vocab_threshold=VOCAB_THRESHOLD,
            vocab_file=VOCAB_FILE,
            vocab_from_file=True,
            download_dir=DOWNLOAD_DIR_TRAIN,
            embedded_imgs=embedded_imgs,
            dataset_path=VAL_DATASET,
            num_imgs=int(NUM_IMG),
        )

    elif EXPERIMENT == "3dshapes":
        # path to pre-saved image features file
        embedded_imgs = torch.load("train_logs/3dshapes_all_ResNet_features_reshaped_all_sq.pt")#torch.cat(( torch.load("3dshapes_all_ResNet_features_reshaped_23000_first.pt"), torch.load("3dshapes_all_ResNet_features_reshaped_240000_first.pt") ), dim = 0)
        transform_train = transforms.Compose([ 
            transforms.ToPILImage(),
            transforms.Resize(IMAGE_SIZE),                   # resize image resolution to 256 (along smaller edge, the other proportionally)
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),               # horizontally flip image with probability=0.5
            transforms.ToTensor(),                           # convert the PIL Image to a tensor
            transforms.Normalize((0.485, 0.456, 0.406),      # normalize image for pre-trained model, tuples for means and std for the three img channels
                         (0.229, 0.224, 0.225))])

        DOWNLOAD_DIR_TRAIN = "../../data"

        data_loader_train = get_loader_3dshapes(
            transform=transform_train,
            mode=MODE,
            batch_size=BATCH_SIZE,
            vocab_threshold=VOCAB_THRESHOLD,
            vocab_file=VOCAB_FILE,
            vocab_from_file=VOCAB_FROM_FILE,
            download_dir=DOWNLOAD_DIR_TRAIN,
            embedded_imgs=embedded_imgs,
        )

        data_loader_val = get_loader_3dshapes(
            transform=transform_train,
            mode="train",
            batch_size=BATCH_SIZE,
            vocab_threshold=VOCAB_THRESHOLD,
            vocab_file=VOCAB_FILE,
            vocab_from_file=VOCAB_FROM_FILE,
            download_dir=DOWNLOAD_DIR_TRAIN,
            embedded_imgs=embedded_imgs,
            dataset_path=VAL_DATASET,
            num_imgs=int(NUM_IMG),
        )
    else:
        raise ValueError(f"Unknown experiment type {EXPERIMENT}")    

    print("NUMBER OF TRAIN IDX: ", len(data_loader_train.dataset.ids))
    print("NUMBER OF VAL IDX: ", len(data_loader_val.dataset.ids))
    print("NUMBER OF VAL caps: ", len(data_loader_val.dataset.caption_lengths))

    # instantiate encoder, decoder, params
    # The size of the vocabulary.
    vocab_size = len(data_loader_train.dataset.vocab)

    print("VOCAB SIZE: ", vocab_size)
    # Initialize the decoder.
    decoder = DecoderRNN(EMBED_SIZE, HIDDEN_SIZE, vocab_size, VISUAL_EMBED_SIZE)
    if SPEAKER_PRETRAINED:
        decoder.load_state_dict(torch.load(SPEAKER_WEIGHTS))

    # Move models to GPU if CUDA is available. 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # encoder.to(device)
    decoder.to(device)

    # Define the loss function. 
    criterion = nn.CrossEntropyLoss().cuda() if torch.cuda.is_available() else nn.CrossEntropyLoss()

    # Specify the learnable parameters of the model.
    params = list(decoder.embed.parameters()) + list(decoder.lstm.parameters()) + list(decoder.linear.parameters()) + list(decoder.project.parameters())

    # Define the optimizer.
    optimizer = torch.optim.Adam(params, lr=0.001, betas=(0.9, 0.999), eps=1e-08)

    # Set the total number of training steps per epoch.
    total_steps = math.floor(len(data_loader_train.dataset.caption_lengths) / data_loader_train.batch_sampler.batch_size)
    print("TOTAL STEPS:", total_steps)

    # training loop
    pretrain_speaker(
        log_file=LOG_FILE,
        num_epochs=EPOCHS,
        total_steps=total_steps,
        data_loader=data_loader_train, 
        data_loader_val=data_loader_val,
        decoder=decoder,
        params=params,
        criterion=criterion,
        optimizer=optimizer,
        weights_path=WEIGHTS_PATH,
        print_every=PRINT_EVERY,
        save_every=SAVE_EVERY,
        train_losses_file=TRAIN_LOSSES_FILE,
        train_metrics_file=TRAIN_METRICS_FILE,
        val_metrics_file=VAL_METRICS_FILE,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # get settings from cmd
    parser.add_argument("-pre_s", "--pretrained_speaker", help = "should speaker be initialized with pretrained weights?", action = "store_true")
    parser.add_argument("-sw", "--speaker_weights", help = "pretrained speaker weights path", default="")
    parser.add_argument("-ep", "--epochs", help = "number of epochs", default=2)
    parser.add_argument("-ex", "--experiment", help = "experiment (3dshapes or coco)", choices=["coco", "3dshapes"] )
    parser.add_argument("-d", "--dataset", help = "path to dataset" )
    parser.add_argument("-dv", "--dataset_val", help = "path to validation dataset")
    parser.add_argument("-ms", "--max_sequence", help = "maximal sequence length")
    parser.add_argument("-b", "--batch_size", help = "batch size", default=64)
    parser.add_argument("-l", "--log_file", help = "logging file")
    parser.add_argument("-vf", "--vocab_file", help = "path to vocab file")
    parser.add_argument("-tl", "--train_losses", help = "path to train losses csv file")
    parser.add_argument("-tm", "--train_metrics", help = "path to train drift metrics csv file")
    parser.add_argument("-vm", "--val_metrics", help = "path to validation drift metrics csv file")
    parser.add_argument("-n_img", "--num_images", help = "number of images to be used for validation")
    parser.add_argument("-wp", "--weights_path", help = "path for saving trained speaker weights")
    
    args = parser.parse_args()
    
    run_pretraining(
        SPEAKER_PRETRAINED=args.pretrained_speaker,
        SPEAKER_WEIGHTS=args.speaker_weights,
        EPOCHS=args.epochs,
        EXPERIMENT=args.experiment,
        DATASET=args.dataset,
        VAL_DATASET=args.dataset_val,
        MAX_SEQUENCE_LEN=args.max_sequence,
        BATCH_SIZE=args.batch_size,
        LOG_FILE=args.log_file,
        VOCAB_FILE=args.vocab_file,
        TRAIN_LOSSES_FILE=args.train_losses,
        TRAIN_METRICS_FILE=args.train_metrics,
        VAL_METRICS_FILE=args.val_metrics,
        NUM_IMG=args.num_images,
        WEIGHTS_PATH=args.weights_path,
    )
    