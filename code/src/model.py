import tensorflow as tf
import os
from utils.build_dataset import make_dataset
from utils.download import maybe_download_and_extract
from utils.encode_captions import TokenizerWrap
from utils.load_records import load_captions_data
from utils.preprocess_images import load_image

# Path to the images
IMAGES_PATH = "Flicker8k_Dataset"

# Desired image dimensions
IMAGE_SIZE = (299, 299)

# Vocabulary size
VOCAB_SIZE = 10000

# Fixed length allowed for any sequence
SEQ_LENGTH = 25

# Dimension for the image embeddings and token embeddings
EMBED_DIM = 512

# Per-layer units in the feed-forward network
FF_DIM = 512

# Other training parameters
BATCH_SIZE = 64
EPOCHS = 30
AUTOTUNE = tf.data.AUTOTUNE
print("beginning main script")

# download data
# think about caching as in example 
DOWNLOAD_DIR_TRAIN = "../../data/train"
DOWNLOAD_DIR_VAL = "../../data/val"

BASE_URL = "http://images.cocodataset.org/"
domains_list = [
    "zips/val2014.zip", 
    # "annotations/annotations_trainval2014.zip",
    # "zips/train2014.zip", 
]
# download data 
for filename in domains_list:
    url = BASE_URL + filename
    print("Downloading ", filename)
    maybe_download_and_extract(
        base_url = BASE_URL,
        filename = filename,
        download_dir = DOWNLOAD_DIR_VAL,
    )
# TODO tab in for iterating over files or make domain a cmd arg
# build records from annotations file
# TODO this is actually unnecessary here
# _, filenames, captions = load_captions_data(
#     download_dir=DOWNLOAD_DIR_VAL, # TODO change to zips or annotations directory?
#     filename="captions_val2014.json",
# )
# print("TYPES: ", type(filenames), " ", type(captions))
# print("executing make daatset")

# build dataset from records, while preprocessing images and text
ds = make_dataset(
    download_dir=DOWNLOAD_DIR_VAL,
    filename="captions", # TODO needs to be filenames of single images 
    is_train=False, 
)
# encode text

# preprocess images

# check if I need a main function
# check if I need to parse cmd args