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
import sacred
import argparse

from dotenv import load_dotenv

load_dotenv()

ex = sacred.Experiment("coco_hyperparameter_search_Lf_only")
usr = os.getenv("MONGO_INITDB_ROOT_USERNAME")
pw = os.getenv("MONGO_INITDB_ROOT_PASSWORD")
db = os.getenv("MONGO_DATABASE")
ex.observers.append(sacred.observers.MongoObserver(
    url=f'mongodb://{usr}:{pw}@localhost:27017/?authMechanism=SCRAM-SHA-1',
    db_name=f"{db}",
))

@ex.capture
def train_reference_game( 
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
    VAL_LOSSES_FILE,
    VAL_METRICS_FILE,
    SPEAKER_PRETRAINED_FILE,
    NUM_IMG,
    PAIRS,
    STRUCTURAL_WEIGHT,
    DECODING_STRATEGY,
    MEAN_BASELINE,
    ENTROPY_WEIGHT,
    **kwargs
):
    """
    Wrapper for conducting reference games wrapped as a sacred experiment, 
    for tracking hyperparameter search results. To be used from with 
    parameters from shell script.
    """
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
    MAX_SEQUENCE_LENGTH = MAX_SEQUENCE_LEN
    # path / name of vocab file
    VOCAB_FILE = "../../data/vocab4000.pkl" #

    # Model Dimensions
    EMBED_SIZE = 512 # dimensionality of word embeddings
    HIDDEN_SIZE = 512 # number of features in hidden state of the LSTM decoder
    VISUAL_EMBED_SIZE = 512 # dimensionality of visual embeddings
    LISTENER_EMBED_SIZE = 512
    # Other training parameters
    BATCH_SIZE = int(BATCH_SIZE)#64
    EPOCHS = int(EPOCHS)#2#20 # number of training epochs
    PRINT_EVERY = 200 # window for printing average loss (steps)
    SAVE_EVERY = 1 # frequency of saving model weights (epochs)
    LOG_FILE = LOG_FILE #'../../data/reference_game_coco_512dim_4000vocab_lf01_log.txt' # name of file with saved training loss and perplexity
    MODE = 'train' # network mode
    WEIGHTS_PATH='../../data/models'
    NUM_VAL_IMGS=3700

    print("Beginning speaker pretraining script...")

    if EXPERIMENT == "coco":

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
            dataset_path=DATASET,
            mode=MODE,
            batch_size=BATCH_SIZE,
            vocab_threshold=VOCAB_THRESHOLD,
            vocab_file=str(VOCAB_FILE),
            vocab_from_file=VOCAB_FROM_FILE,
            download_dir=DOWNLOAD_DIR_TRAIN,
            embedded_imgs=embedded_imgs,
            num_imgs=int(NUM_IMG),
            pairs=PAIRS,
        )
        data_loader_val = get_loader(
            transform=transform_train,
            dataset_path=VAL_DATASET,
            mode="train",
            batch_size=BATCH_SIZE,
            vocab_threshold=VOCAB_THRESHOLD,
            vocab_file=VOCAB_FILE,
            vocab_from_file=True,
            download_dir=DOWNLOAD_DIR_TRAIN,
            embedded_imgs=embedded_imgs,
            num_imgs=int(250),
            pairs=PAIRS,
        )
        print("NUMBER OF VALIDATION IMAGES ", len(data_loader_val.dataset.ids))
    else:
        # 3dshapes
        transform_train = transforms.Compose([ 
            transforms.ToPILImage(),
            transforms.Resize(IMAGE_SIZE),                   # resize image resolution to 256 (along smaller edge, the other proportionally)
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),               # horizontally flip image with probability=0.5
            transforms.ToTensor(),                           # convert the PIL Image to a tensor
            transforms.Normalize((0.485, 0.456, 0.406),      # normalize image for pre-trained model, tuples for means and std for the three img channels
                                (0.229, 0.224, 0.225))])
        DOWNLOAD_DIR_TRAIN = "../../data"
        # path to pre-saved image features file
        embedded_imgs = torch.load("3dshapes_all_ResNet_features_reshaped_all_sq.pt")

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
            mode=MODE,
            batch_size=BATCH_SIZE,
            vocab_threshold=VOCAB_THRESHOLD,
            vocab_file=VOCAB_FILE,
            vocab_from_file=VOCAB_FROM_FILE,
            download_dir=DOWNLOAD_DIR_TRAIN,
            embedded_imgs=embedded_imgs,
        )
    

    print("NUMBER OF TRAIN IDX: ", len(data_loader_train.dataset.ids))
    # print("NUMBER OF VAL IDX: ", len(data_loader_val.dataset.ids))
    # print("NUMBER OF VAL caps: ", len(data_loader_val.dataset.caption_lengths))

    # instantiate encoder, decoder, params
    # The size of the vocabulary.
    vocab_size = len(data_loader_train.dataset.vocab)

    print("VOCAB SIZE: ", vocab_size)
    # Initialize the encoder and decoder.
    # Encoder projects the concatenation of the two images to the concatenation of the desired visual embedding size 
    # speaker_encoder = EncoderMLP(2048, VISUAL_EMBED_SIZE)
    listener_encoder = ListenerEncoderCNN(LISTENER_EMBED_SIZE)
    # listener_encoder.load_state_dict(torch.load("models/listener-encoder-wPretrained-vocab4000-metrics-1.pkl"))
    # print("Model summaries:")
    # print(listener_encoder.summary())

    print("Listener encoder requires grad: ", sum(p.numel() for p in listener_encoder.parameters() if p.requires_grad) )


    speaker_decoder = DecoderRNN(EMBED_SIZE, HIDDEN_SIZE, vocab_size, VISUAL_EMBED_SIZE)
    listener_rnn = ListenerEncoderRNN(LISTENER_EMBED_SIZE, HIDDEN_SIZE, vocab_size)
    print("Listener RNN requires grad: ", sum(p.numel() for p in listener_rnn.parameters() if p.requires_grad) )
    print("Speaker RNN requires grad: ", sum(p.numel() for p in speaker_decoder.parameters() if p.requires_grad) )

    speaker_decoder.load_state_dict(torch.load(str(SPEAKER_PRETRAINED_FILE))) # "models/decoder-noEnc-prepend-512dim-4000vocab-rs1234-wEmb-cont-7.pkl"
    # listener_rnn.load_state_dict(torch.load("models/listener-rnn-wPretrained-vocab4000-metrics-1.pkl"))
    # Move models to GPU if CUDA is available. 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # speaker_encoder.to(device)
    speaker_decoder.to(device)
    listener_encoder.to(device)
    listener_rnn.to(device)

    # Define the loss function. 
    criterion = nn.CrossEntropyLoss().cuda() if torch.cuda.is_available() else nn.CrossEntropyLoss()

    # Set the total number of training steps per epoch.
    total_steps = math.ceil(len(data_loader_train.dataset.caption_lengths) / data_loader_train.batch_sampler.batch_size)
    print("TOTAL STEPS:", total_steps)

    # training loop
    loss_speaker_train, loss_str_train, loss_f_train, ppl_train, loss_l_train,\
        acc_train, str_drift_pred, str_drift_true, sem_drift_pred, sem_drift_true,\
        disc_overlaps, cont_overlaps, img_sims, epochs_metrics, loss_str_val_all, ppl_val_all,\
        loss_val_avg, ppl_val_avg, val_steps = play_game(
            log_file=LOG_FILE,
            num_epochs=EPOCHS,
            total_steps=total_steps,
            data_loader=data_loader_train, 
            data_loader_val=data_loader_val,
            speaker_decoder=speaker_decoder,
            listener_encoder=listener_encoder, 
            listener_rnn=listener_rnn,
            criterion=criterion,
            weights_path=WEIGHTS_PATH,
            print_every=PRINT_EVERY,
            save_every=SAVE_EVERY,
            train_losses_file=TRAIN_LOSSES_FILE,
            train_metrics_file=TRAIN_METRICS_FILE,
            val_losses_file=VAL_LOSSES_FILE,
            val_metrics_file=VAL_METRICS_FILE,
            experiment=EXPERIMENT,
            lambda_s=float(STRUCTURAL_WEIGHT),
            pretrained_decoder_file=SPEAKER_PRETRAINED_FILE,
            decoding_strategy=DECODING_STRATEGY,
            mean_baseline=MEAN_BASELINE,
            entropy_weight=float(ENTROPY_WEIGHT),
        )

    # dump training stats to sacred db
    for i in range(len(loss_speaker_train)):
        ex.log_scalar("speaker_loss_train", value=loss_speaker_train[i], step=i)
        ex.log_scalar("loss_structural_train", value=loss_str_train[i], step=i)
        ex.log_scalar("loss_functional_train", value=loss_f_train[i], step=i)
        ex.log_scalar("perplexities_train", value=ppl_train[i], step=i)
        ex.log_scalar("loss_listener_train", value=loss_l_train[i], step=i)
        ex.log_scalar("listener_acc_train", value=acc_train[i], step=i)
        
    # dump val stats to sacred db
    for i in range(len(val_steps)):
        ex.log_scalar("structural_drift_pred", value=str_drift_pred[i], step=i)
        ex.log_scalar("structural_drift_true", value=str_drift_true[i], step=i)
        ex.log_scalar("semantic_drift_pred", value=sem_drift_pred[i], step=i)
        ex.log_scalar("semantic_drift_true", value=sem_drift_true[i], step=i)
        ex.log_scalar("discrete_overlaps", value=disc_overlaps[i], step=i)
        ex.log_scalar("continuous_overlaps", value=cont_overlaps[i], step=i)
        ex.log_scalar("image_similarities_val", value=img_sims[i], step=i)
        ex.log_scalar("epochs_val", value=epochs_metrics[i], step=i)
        ex.log_scalar("loss_structural_val", value=loss_str_val_all[i], step=i)
        ex.log_scalar("perplexities_val", value=ppl_val_all[i], step=i)
        ex.log_scalar("val_steps", value=val_steps[i], step=i)
    # dump evaluation round level averages
    for i in range(len(loss_val_avg)):
        ex.log_scalar("structural_drift_val_epoch_avg", value=loss_val_avg[i], step=i)
        ex.log_scalar("ppl_val_epoch_avg", value=ppl_val_avg[i], step=i)
            
    
@ex.main 
def run(_config):
    train_reference_game(**_config)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # flag for conducting hyperparams expt or just doing test mode
    parser.add_argument("-deb", "--debug_mode", help = "should the script be run in debug mode?", action="store_true")

    # general utils
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
    parser.add_argument("-vl", "--val_losses", help = "path to validation losses csv file")
    parser.add_argument("-vm", "--val_metrics", help = "path to validation drift metrics csv file")
    parser.add_argument("-s_pre", "--speaker_pretrained", help = "path to pretrained speaker decoder model")
    parser.add_argument("-n_img", "--num_images", help = "number of images to be used")
    parser.add_argument("-p", "--pairs", help = "type of target/distractor pairs (similar, random)", choices=["random", "similar"])
    

    # grid search specific parameters
    parser.add_argument("-l_s", "--lambda_structural", help = "weight of structural loss")
    parser.add_argument("-str", "--decoding_strategy", help = "decoding strategy for speaker", choices = ["pure", "greedy", "exp"])
    parser.add_argument("-mb", "--mean_baseline", help = "use mean baseline subtraction?", action="store_true")
    parser.add_argument("-entr", "--entropy_weight", help = "weight of entropy regularization of REINFORCE")
    parser.add_argument("-l_f", "--lambda_functional", help = "weight of functional loss")
    
    
    args = parser.parse_args()

    if not args.debug_mode:

        @ex.config
        def config():
            EPOCHS=args.epochs
            EXPERIMENT=args.experiment
            DATASET=args.dataset
            VAL_DATASET=args.dataset_val
            MAX_SEQUENCE_LEN=args.max_sequence
            BATCH_SIZE=args.batch_size
            LOG_FILE=args.log_file
            VOCAB_FILE=args.vocab_file
            TRAIN_LOSSES_FILE=args.train_losses
            TRAIN_METRICS_FILE=args.train_metrics
            VAL_LOSSES_FILE=args.val_losses
            VAL_METRICS_FILE=args.val_metrics
            SPEAKER_PRETRAINED_FILE=args.speaker_pretrained
            NUM_IMG=args.num_images
            PAIRS=args.pairs
            STRUCTURAL_WEIGHT=args.lambda_structural
            DECODING_STRATEGY=args.decoding_strategy
            MEAN_BASELINE=args.mean_baseline
            ENTROPY_WEIGHT=args.entropy_weight
        
        ex.run()
    else:
        # just execute train loop
        train_reference_game(
            EPOCHS=2,
            EXPERIMENT="coco",
            DATASET="train_logs/ref-game_img_IDs_15000_coco_lf01.pt",
            VAL_DATASET="notebooks/val_split_IDs_from_COCO_train_tensor.pt",
            MAX_SEQUENCE_LEN=15,
            BATCH_SIZE=64,
            LOG_FILE="../../data/debug_log_file.txt",
            VOCAB_FILE="../../data/vocab4000.pkl",
            TRAIN_LOSSES_FILE="",
            TRAIN_METRICS_FILE="",
            VAL_LOSSES_FILE="",
            VAL_METRICS_FILE="",
            SPEAKER_PRETRAINED_FILE="models/decoder-noEnc-prepend-512dim-4000vocab-rs1234-wEmb-cont-7.pkl",
            NUM_IMG=1000,
            PAIRS="random",
            STRUCTURAL_WEIGHT=0.9,
            DECODING_STRATEGY="pure",
            MEAN_BASELINE=False,
            ENTROPY_WEIGHT=0.1,
        )