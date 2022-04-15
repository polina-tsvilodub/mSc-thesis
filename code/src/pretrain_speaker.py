# import tensorflow as tf
import os
from utils.build_dataset import make_dataset, get_loader
from utils.download import maybe_download_and_extract
from utils.train import pretrain_speaker
import torch
from torchvision import transforms
import torch.nn as nn
from agents.speaker import SpeakerEncoderCNN, DecoderRNN
import math


# set random seed
torch.manual_seed(42)

#######
# HYPERPARAMS 
########

# Desired image dimensions
IMAGE_SIZE = 256

# Vocabulary parameters
VOCAB_SIZE = 10000
VOCAB_THRESHOLD = 1 # minimum word count threshold
VOCAB_FROM_FILE = True # if True, load existing vocab file
VOACAB_FROM_PRETRAINED = False
# Fixed length allowed for any sequence
MAX_SEQUENCE_LENGTH = 25
# path / name of vocab file
VOCAB_FILE = "../../data/vocab.pkl"

# Model Dimensions
EMBED_SIZE = 512 # dimensionality of image and word embeddings, needs to match because they are concatenated
HIDDEN_SIZE = 512 # number of features in hidden state of the LSTM decoder

# Other training parameters
BATCH_SIZE = 64
EPOCHS = 30 # number of training epochs
PRINT_EVERY = 500 # window for printing average loss (steps)
SAVE_EVERY = 1 # frequency of saving model weights (epochs)
LOG_FILE = '../../data/speaker_pretraining_log.txt' # name of file with saved training loss and perplexity
MODE= 'train' # network mode
WEIGHTS_PATH='../../data/models'

# data download params
DOWNLOAD_DIR_TRAIN = "../../data/train"
DOWNLOAD_DIR_VAL = "../../data/val"

BASE_URL = "http://images.cocodataset.org/"
domains_list = {
    DOWNLOAD_DIR_VAL: "zips/val2014.zip", 
    DOWNLOAD_DIR_TRAIN: ["annotations/annotations_trainval2014.zip",
    "zips/train2014.zip"], 
}

#########

print("Beginning speaker pretraining script...")

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
    transforms.Pad(32),
    transforms.RandomCrop(IMAGE_SIZE),
    transforms.RandomHorizontalFlip(),               # horizontally flip image with probability=0.5
    transforms.ToTensor(),                           # convert the PIL Image to a tensor
    # TODO check params
    transforms.Normalize((0.485, 0.456, 0.406),      # normalize image for pre-trained model, tuples for means and std for the three img channels
                         (0.229, 0.224, 0.225))])

# TODO renormalize image again for output if necessary, and think if these transforms need to be the same for speaker and listener functional training

# Build data loader, allowing to iterate over records from annotations file
data_loader = get_loader(
    transform=transform_train,
    mode=MODE,
    batch_size=BATCH_SIZE,
    vocab_threshold=VOCAB_THRESHOLD,
    vocab_file=VOCAB_FILE,
    vocab_from_file=VOCAB_FROM_FILE,
    download_dir=DOWNLOAD_DIR_TRAIN,
)

# instantiate encoder, decoder, params
# The size of the vocabulary.
vocab_size = len(data_loader.dataset.vocab)

# Initialize the encoder and decoder. 
encoder = SpeakerEncoderCNN(EMBED_SIZE)
decoder = DecoderRNN(EMBED_SIZE, HIDDEN_SIZE, vocab_size)

# Move models to GPU if CUDA is available. 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
encoder.to(device)
decoder.to(device)

# Define the loss function. 
criterion = nn.CrossEntropyLoss().cuda() if torch.cuda.is_available() else nn.CrossEntropyLoss()

# Specify the learnable parameters of the model.
params = list(decoder.lstm.parameters()) + list(decoder.linear.parameters()) + list(encoder.embed.parameters()) + list(encoder.batch.parameters())

# Define the optimizer.
optimizer = torch.optim.Adam(params, lr=0.001, betas=(0.9, 0.999), eps=1e-08)

# Set the total number of training steps per epoch.
total_steps = math.ceil(len(data_loader.dataset.caption_lengths) / data_loader.batch_sampler.batch_size)
print("TORAL STEPS:", total_steps)

# training loop
pretrain_speaker(
    log_file=LOG_FILE,
    num_epochs=EPOCHS,
    total_steps=total_steps,
    data_loader=data_loader, 
    encoder=encoder,
    decoder=decoder,
    params=params,
    criterion=criterion,
    optimizer=optimizer,
    weights_path=WEIGHTS_PATH,
    print_every=PRINT_EVERY,
    save_every=SAVE_EVERY,
)
# dump training stats and model 

# check if I need a main function
# check if I need to parse cmd args