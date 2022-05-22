# next to drift metrics, track somehow how the targets and distractors are paired
# i.e., track ther (dis)similarity, to be able to make to caption granularity comparison to the second experiment

import os
from utils.build_dataset import make_dataset, get_loader
from utils.download import maybe_download_and_extract
from utils.train import pretrain_speaker
from utils.early_stopping import EarlyStopping
from reference_game_utils.train import play_game
import torch
from torchvision import transforms
import torch.nn as nn
from agents.speaker import DecoderRNN
from agents.resnet_encoder import EncoderMLP
from agents.listener import ListenerEncoderRNN, ListenerEncoderCNN
import math
import random


# set random seed
torch.manual_seed(42)
random.seed(42)

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
EMBED_SIZE = 1024 # dimensionality of word embeddings
HIDDEN_SIZE = 512 # number of features in hidden state of the LSTM decoder
VISUAL_EMBED_SIZE = 512 # dimensionality of visual embeddings
LISTENER_EMBED_SIZE = 512
# Other training parameters
BATCH_SIZE = 64
EPOCHS = 5#20 # number of training epochs
PRINT_EVERY = 200 # window for printing average loss (steps)
SAVE_EVERY = 1 # frequency of saving model weights (epochs)
LOG_FILE = '../../data/reference_game_token0_noEnc_1024dim_4000vocab_wFeatures_metrics_log.txt' # name of file with saved training loss and perplexity
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

# path to pre-saved image features file
embedded_imgs = torch.load("COCO_train_ResNet_features_reshaped.pt")
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
    embedded_imgs=embedded_imgs,
)

data_loader_val = get_loader(
    transform=transform_train,
    mode="val",
    batch_size=BATCH_SIZE,
    vocab_threshold=VOCAB_THRESHOLD,
    vocab_file=VOCAB_FILE,
    vocab_from_file=True,
    download_dir=DOWNLOAD_DIR_VAL,
    embedded_imgs=embedded_imgs,
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
# speaker_encoder = EncoderMLP(2048, VISUAL_EMBED_SIZE)
listener_encoder = ListenerEncoderCNN(LISTENER_EMBED_SIZE)
listener_encoder.load_state_dict(torch.load("models/listener-encoder-noEnc-token0-vocab4000-1.pkl"))
# print("Model summaries:")

# print("Listener MLP requires grad: ", list(filter(lambda p: p.requires_grad, listener_encoder.parameters())))

speaker_decoder = DecoderRNN(EMBED_SIZE, HIDDEN_SIZE, vocab_size, VISUAL_EMBED_SIZE)
listener_rnn = ListenerEncoderRNN(LISTENER_EMBED_SIZE, HIDDEN_SIZE, vocab_size)

speaker_decoder.load_state_dict(torch.load("models/speaker-decoder-noEnc-token0-vocab4000-1.pkl"))
listener_rnn.load_state_dict(torch.load("models/listener-rnn-noEnc-token0-vocab4000-1.pkl"))
# Move models to GPU if CUDA is available. 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# speaker_encoder.to(device)
speaker_decoder.to(device)
listener_encoder.to(device)
listener_rnn.to(device)

# Define the loss function. 
criterion = nn.CrossEntropyLoss().cuda() if torch.cuda.is_available() else nn.CrossEntropyLoss()

# Specify the learnable parameters of the model.
# params = list(decoder.lstm.parameters()) + list(decoder.linear.parameters()) + list(encoder.embed.parameters())

# print("Encoder MLP params: ", list(encoder.embed.parameters()))
# print("Encoder CNN params: ", list(old_encoder.embed.parameters()))
# Define the optimizer.
# optimizer = torch.optim.Adam(params, lr=0.001, betas=(0.9, 0.999), eps=1e-08)

# Set the total number of training steps per epoch.
total_steps = math.ceil(len(data_loader_train.dataset.caption_lengths) / data_loader_train.batch_sampler.batch_size)
print("TOTAL STEPS:", total_steps)

# training loop
play_game(
    log_file=LOG_FILE,
    num_epochs=EPOCHS,
    total_steps=total_steps,
    data_loader=data_loader_train, 
    data_loader_val=data_loader_val,
    # speaker_encoder=speaker_encoder,
    speaker_decoder=speaker_decoder,
    listener_encoder=listener_encoder, 
    listener_rnn=listener_rnn,
    criterion=criterion,
    weights_path=WEIGHTS_PATH,
    print_every=PRINT_EVERY,
    save_every=SAVE_EVERY,
)
# dump training stats and model 

# check if I need a main function
# check if I need to parse cmd args