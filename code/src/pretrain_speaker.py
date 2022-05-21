# import tensorflow as tf
import os
from utils.build_dataset import make_dataset, get_loader
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


# set random seed
torch.manual_seed(1234)
random.seed(1234)

#######
# HYPERPARAMS 
########

# Desired image dimensions
IMAGE_SIZE = 256

# Vocabulary parameters
VOCAB_THRESHOLD = 25 # minimum word count threshold
VOCAB_FROM_FILE = True # if True, load existing vocab file
VOCAB_FROM_PRETRAINED = False
# Fixed length allowed for any sequence
MAX_SEQUENCE_LENGTH = 15
# path / name of vocab file
VOCAB_FILE = "../../data/vocab4000.pkl"

# Model Dimensions
EMBED_SIZE = 1024 # 1024 # dimensionality of word embeddings
HIDDEN_SIZE = 512 # number of features in hidden state of the LSTM decoder
VISUAL_EMBED_SIZE = 512 # dimensionality of visual embeddings

# Other training parameters
BATCH_SIZE = 64
EPOCHS = 20 # number of training epochs
PRINT_EVERY = 200 # window for printing average loss (steps)
SAVE_EVERY = 1 # frequency of saving model weights (epochs)
LOG_FILE = '../../data/CCE_pretraining_speaker_noEnc_prepend_1024dim_4000vocab_rs1234_cont_log.txt' # name of file with saved training loss and perplexity
MODE= 'train' # network mode
WEIGHTS_PATH='../../data/models'
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
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(),               # horizontally flip image with probability=0.5
    transforms.ToTensor(),                           # convert the PIL Image to a tensor
    transforms.Normalize((0.485, 0.456, 0.406),      # normalize image for pre-trained model, tuples for means and std for the three img channels
                         (0.229, 0.224, 0.225))])

# TODO renormalize image again for output if necessary, and think if these transforms need to be the same for speaker and listener functional training

# Build data loader, allowing to iterate over records from annotations file
data_loader_train = get_loader(
    transform=transform_train,
    mode=MODE,
    batch_size=BATCH_SIZE,
    vocab_threshold=VOCAB_THRESHOLD,
    vocab_file=VOCAB_FILE,
    vocab_from_file=VOCAB_FROM_FILE,
    download_dir=DOWNLOAD_DIR_TRAIN,
)

data_loader_val = get_loader(
    transform=transform_train,
    mode="val",
    batch_size=BATCH_SIZE,
    vocab_threshold=VOCAB_THRESHOLD,
    vocab_file=VOCAB_FILE,
    vocab_from_file=True,
    download_dir=DOWNLOAD_DIR_VAL,
)
# truncate the val split
data_loader_val.dataset.ids = torch.load("pretrain_val_img_IDs_2imgs_main.pt").tolist()#data_loader_val.dataset.ids[:NUM_VAL_IMGS]
data_loader_val.dataset.caption_lengths = data_loader_val.dataset.caption_lengths[:NUM_VAL_IMGS]
# save
# torch.save(torch.tensor(data_loader_val.dataset.ids), "pretrain_val_img_IDs_2imgs_main.pt")

print("NUMBER OF TRAIN IDX: ", len(data_loader_train.dataset.ids))
print("NUMBER OF VAL IDX: ", len(data_loader_val.dataset.ids))
print("NUMBER OF VAL caps: ", len(data_loader_val.dataset.caption_lengths))

# instantiate encoder, decoder, params
# The size of the vocabulary.
vocab_size = len(data_loader_train.dataset.vocab)

print("VOCAB SIZE: ", vocab_size)
# Initialize the encoder and decoder.
# Encoder projects the concatenation of the two images to the concatenation of the desired visual embedding size 
# encoder = EncoderMLP(2048, VISUAL_EMBED_SIZE)
# old_encoder = EncoderCNN(VISUAL_EMBED_SIZE)
# print("State dict keys: ", torch.load_state_dict("models/encoder-2imgs-1.pkl").keys())
# pretrained_dict = torch.load("models/encoder-2imgs-1.pkl")
# encoder_dict = encoder.state_dict()
# # filter out unnecessary keys
# pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in encoder_dict}
# # overwrite entries in the existing state dict
# encoder_dict.update(pretrained_dict) 
# # load the new state dict
# encoder.load_state_dict(pretrained_dict)
# print("LOADED ENCODER WEIGHTS!")
# encoder.load_state_dict(torch.load("models/encoder-2imgs-1024dim-2000vocab-1.pkl"))
decoder = DecoderRNN(EMBED_SIZE, HIDDEN_SIZE, vocab_size, VISUAL_EMBED_SIZE)
# decoder.load_state_dict(torch.load("models/decoder-noEnc-prepend-1024dim-4000vocab-rs1234-2.pkl"))

# Move models to GPU if CUDA is available. 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# encoder.to(device)
decoder.to(device)

# Define the loss function. 
criterion = nn.CrossEntropyLoss().cuda() if torch.cuda.is_available() else nn.CrossEntropyLoss()

# Specify the learnable parameters of the model.
params = list(decoder.lstm.parameters()) + list(decoder.linear.parameters()) + list(decoder.project.parameters())

# print("Encoder MLP params: ", list(encoder.embed.parameters()))
# print("Encoder CNN params: ", list(old_encoder.embed.parameters()))
# Define the optimizer.
optimizer = torch.optim.Adam(params, lr=0.001, betas=(0.9, 0.999), eps=1e-08)

# Set the total number of training steps per epoch.
total_steps = math.ceil(len(data_loader_train.dataset.caption_lengths) / data_loader_train.batch_sampler.batch_size)
print("TOTAL STEPS:", total_steps)

# training loop
pretrain_speaker(
    log_file=LOG_FILE,
    num_epochs=EPOCHS,
    total_steps=total_steps,
    data_loader=data_loader_train, 
    data_loader_val=data_loader_val,
    # encoder=old_encoder,
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