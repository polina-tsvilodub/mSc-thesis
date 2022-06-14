from agents.resnet_encoder import ResNetPreprocessor
from utils.build_dataset import get_loader, get_loader_3dshapes
import torch
from torchvision import transforms
from tqdm import tqdm

def extract_and_save_resnet_features(
    file: str,
    mode: str,
    out_file: str,
):
    """
    Given a file of COCO caption IDs (train or val), extract image features and save them to a torch file.
    Arguments:
    ---------
    file: str
        Path to COCO annotations file.
    mode: str
        Train or val.  
    out_file: str
        Path to out file for extracted features.      
    """
    torch.manual_seed(42)

    transform_train = transforms.Compose([ 
        transforms.ToPILImage(),                         # necessary for 3dshapes images
        transforms.Resize(256),                          # smaller edge of image resized to 256
        transforms.RandomCrop(224),                      # get 224x224 crop from random location
        transforms.RandomHorizontalFlip(),               # horizontally flip image with probability=0.5
        transforms.ToTensor(),                           # convert the PIL Image to a tensor
        transforms.Normalize((0.485, 0.456, 0.406),      # normalize image for pre-trained model
                            (0.229, 0.224, 0.225))]
    )

    data_loader = get_loader_3dshapes(
        transform=transform_train,
        mode=mode,
        batch_size=1,
        vocab_threshold=11,
        vocab_from_file=True,
        download_dir=file,
    )

    # instanitate ResNet
    resnet = ResNetPreprocessor()
    resnet.eval()

    # list for saving
    preprocessed_images = []

    print("Number of imgs for feature extraction: ", len(data_loader.dataset))
    for i in tqdm(range(len(data_loader.dataset))): # .ids for COCO
        resnet.eval()
        # Obtain the batch.
        new_sampler = torch.utils.data.sampler.SubsetRandomSampler(indices=[(i, 0)]) # tuple index, since __getitem__ expects two indices for target and distractor
        data_loader.batch_sampler.sampler = new_sampler
        # get the images from the data loader
        targets, distractors, target_captions = next(iter(data_loader))
        # pass them through the resnet
        features = resnet(targets)
        preprocessed_images.append(features)
        
        if i % 5000 == 0 :
            print("Saving at ", i)
            preprocessed_images_tensor = torch.stack(preprocessed_images)
            torch.save(preprocessed_images_tensor,  "3dshapes_all_ResNet_features_reshaped_" + str(i) + ".pt")
        
        # save all images then
    preprocessed_images_tensor = torch.stack(preprocessed_images)
    print("Finished extracting!")
    print(preprocessed_images_tensor.shape)
    torch.save(preprocessed_images_tensor, out_file)

if __name__ == "__main__":
    extract_and_save_resnet_features(
        "../../data",
        "train",
        "3dshapes_all_ResNet_features_reshaped.pt",
    )

