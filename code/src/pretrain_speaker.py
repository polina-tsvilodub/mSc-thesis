# import tensorflow as tf
import os
from utils.build_dataset import make_dataset, get_loader
from utils.download import maybe_download_and_extract
import torch
from torchvision import transforms
import torch.nn as nn
from agents.speaker import SpeakerEncoderCNN, DecoderRNN
import math
# from utils.encode_captions import TokenizerWrap
# from utils.load_records import load_captions_data
# from utils.preprocess_images import load_image


# Desired image dimensions
IMAGE_SIZE = (299, 299)

# Vocabulary size
VOCAB_SIZE = 10000

# Fixed length allowed for any sequence
# SEQ_LENGTH = 25

# Dimension for the image embeddings and token embeddings
EMBED_SIZE = 512

# Per-layer units in the feed-forward network
FF_DIM = 512

# Other training parameters
BATCH_SIZE = 64
EPOCHS = 30
# AUTOTUNE = tf.data.AUTOTUNE
print("beginning main script")

# download data
# think about caching as in example 
DOWNLOAD_DIR_TRAIN = "../../data/train"
DOWNLOAD_DIR_VAL = "../../data/val"

BASE_URL = "http://images.cocodataset.org/"
domains_list = [
    # "zips/val2014.zip", 
    "annotations/annotations_trainval2014.zip",
    "zips/train2014.zip", 
]
# download data 
for filename in domains_list:
    url = BASE_URL + filename
    print("Downloading ", filename)
    maybe_download_and_extract(
        base_url = BASE_URL,
        filename = filename,
        download_dir = DOWNLOAD_DIR_TRAIN,
    )
# TODO tab in for iterating over files or make domain a cmd arg
# build records from annotations file
# TODO this is actually unnecessary here

vocab_threshold = 1        # minimum word count threshold
vocab_from_file = True    # if True, load existing vocab file
# embed_size = 512           # dimensionality of image and word embeddings
hidden_size = 512          # number of features in hidden state of the RNN decoder
num_epochs = 1             # number of training epochs (1 for testing)
save_every = 1             # determines frequency of saving model weights
print_every = 200          # determines window for printing average loss
log_file = 'training_log.txt'       # name of file with saved training loss and perplexity

# (Optional) TODO #2: Amend the image transform below.
transform_train = transforms.Compose([ 
    transforms.Resize(256),                          # smaller edge of image resized to 256
    transforms.RandomCrop(224),                      # get 224x224 crop from random location
    transforms.RandomHorizontalFlip(),               # horizontally flip image with probability=0.5
    transforms.ToTensor(),                           # convert the PIL Image to a tensor
    transforms.Normalize((0.485, 0.456, 0.406),      # normalize image for pre-trained model
                         (0.229, 0.224, 0.225))])

# Build data loader.
data_loader = get_loader(
    transform=transform_train,
    mode='train',
    batch_size=BATCH_SIZE,
    vocab_threshold=vocab_threshold,
    vocab_file="../../data/vocab.pkl",
    vocab_from_file=vocab_from_file,
    download_dir=DOWNLOAD_DIR_TRAIN,
)

# instantiate encoder, decoder, params
# The size of the vocabulary.
vocab_size = len(data_loader.dataset.vocab)

# Initialize the encoder and decoder. 
encoder = SpeakerEncoderCNN(EMBED_SIZE)
decoder = DecoderRNN(EMBED_SIZE, hidden_size, vocab_size)

# Move models to GPU if CUDA is available. 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
encoder.to(device)
decoder.to(device)

# Define the loss function. 
criterion = nn.CrossEntropyLoss().cuda() if torch.cuda.is_available() else nn.CrossEntropyLoss()

# TODO #3: Specify the learnable parameters of the model.
params = list(decoder.lstm.parameters()) + list(decoder.linear.parameters()) + list(encoder.embed.parameters()) + list(encoder.batch.parameters())

# TODO #4: Define the optimizer.
optimizer = torch.optim.Adam(params, lr=0.001, betas=(0.9, 0.999), eps=1e-08)
# optimizer = torch.optim.Adam(params, lr=0.01, betas=(0.9, 0.999), eps=1e-08)
# optimizer = torch.optim.RMSprop(params, lr=0.01, alpha=0.99, eps=1e-08)

# Set the total number of training steps per epoch.
total_step = math.ceil(len(data_loader.dataset.caption_lengths) / data_loader.batch_sampler.batch_size)
print("TORAL STEPS:", total_step)

# training loop

# dump training stats and model 

# _, filenames, captions = load_captions_data(
#     download_dir=DOWNLOAD_DIR_VAL, # TODO change to zips or annotations directory?
#     filename="captions_val2014.json",
# )
# print("TYPES: ", type(filenames), " ", type(captions))
# print("executing make daatset")

# build dataset from records, while preprocessing images and text
# ds = make_dataset(
#     download_dir=DOWNLOAD_DIR_TRAIN,
#     filename="captions", # TODO needs to be filenames of single images 
#     is_train=False, 
# )
# encode text

# preprocess images

# check if I need a main function
# check if I need to parse cmd args