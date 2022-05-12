import os 
import json
import numpy as np
from random import shuffle
from . import vocabulary
from pycocotools.coco import COCO
import torch
from torch.utils.data import Dataset
from torchtext.data import get_tokenizer
from PIL import Image
from tqdm import tqdm

# build a batch generator which takes ones caption out of the five at random
# TODO add a bool for whether the imgs should be sorted into categories
# and if so, do that on the self.ids  

class COCOCaptionsDataset(Dataset):
    """
    Custom class for preprocessing datapoints and sampling a random caption per image.
    """
    def __init__(
            self, 
            file, 
            download_dir, 
            img_transform, 
            text_transform, 
            batch_size, 
            mode, 
            vocab_threshold, 
            vocab_file, 
            start_token, 
            end_token,
            unk_token,
            pad_token, 
            vocab_from_file, 
            vocab_from_pretrained=False,
            max_sequence_length=0,
            categorize_imgs=False, # flag whether the images need to be sorted
        ):
        self.transform = img_transform
        self.mode = mode # train or test or val
        self.batch_size = batch_size
        self.vocab = vocabulary.Vocabulary(vocab_threshold, vocab_file, 
                                start_token, end_token, unk_token, 
                                file, pad_token, vocab_from_file, vocab_from_pretrained) 
        self.max_sequence_length = max_sequence_length
        self.pad_token=pad_token
        # some distinctions below for Train and test mode
        if mode == "train":
            self.image_dir = os.path.join(download_dir, "train2014") # download_dir needs to be data/train/ then 
            self.coco = COCO(file) # os.path.join(download_dir, file)
            _ids = list(self.coco.anns.keys())
            shuffle(_ids)
            # retrieve a subset of images for pretraining
            self.ids = list(self.coco.anns.keys()) #torch.load("pretrain_img_IDs_2imgs.pt").tolist()#_ids[:70000]
            # set the image IDs for validation during early stopping to avoid overlapping images
            self.ids_val = torch.load("pretrain_val_img_IDs_2imgs.pt").tolist() #_ids[70000:73700]
            print('Obtaining caption lengths...')
            tokenizer = get_tokenizer("basic_english") # nltk.tokenize.word_tokenize(str(self.coco.anns[self.ids[index]]['caption']).lower())
            all_tokens = [tokenizer(str(self.coco.anns[self.ids[index]]['caption']).lower()) for index in tqdm(np.arange(len(self.ids)))] 
            all_tokens_val = [tokenizer(str(self.coco.anns[self.ids_val[index]]['caption']).lower()) for index in tqdm(np.arange(len(self.ids_val)))] 
            self.caption_lengths = [len(token) for token in all_tokens]
            self.caption_lengths_val = [len(token) for token in all_tokens_val] 
            # print pretraining IDs for later separation from functional training
            # save used indices to torch file
            torch.save(torch.tensor(self.ids), "pretrain_img_IDs_2imgs_1024dim.pt")
            torch.save(torch.tensor(self.ids_val), "pretrain_val_img_IDs_2imgs_1024dim.pt")

        elif mode == "val":
            self.image_dir = os.path.join(download_dir, "val2014")
            self.coco = COCO(file) # os.path.join(download_dir, file)
            self.ids = list(self.coco.anns.keys())#[:1000]
            print('Obtaining caption lengths...')
            tokenizer = get_tokenizer("basic_english") # nltk.tokenize.word_tokenize(str(self.coco.anns[self.ids[index]]['caption']).lower())
            all_tokens = [tokenizer(str(self.coco.anns[self.ids[index]]['caption']).lower()) for index in tqdm(np.arange(len(self.ids)))] 
            self.caption_lengths = [len(token) for token in all_tokens]
            
        else:
            # no annotations here 
            test_info = json.loads(open(file).read()) # os.path.join(download_dir, file)
            self.paths = [item['file_name'] for item in test_info['images']]
    
    def __len__(self):
        if self.mode != "test":
            return len(self.ids) 
        else:
            return len(self.paths)
    
    def __getitem__(self, idx):
        """
        Return an image-caption tuple. A random caption per images is chosen since the dataset maps captions onto images.
        
        Arguments:
        -------
        idx: int
            Index of the item to be returned.
        Returns:
        -----
        image: torch.tensor((3,224,224))
        caption: torch.tensor((len_caption))
        """
        
        # obtain image and caption if in training mode
        if self.mode != 'test':
            # get target and distractor indices
            target_idx = idx[0]
            distractor_idx = idx[1]
            
            ann_id = self.ids[target_idx]
            target_caption = self.coco.anns[ann_id]['caption']
            img_id = self.coco.anns[ann_id]['image_id']
            target_path = self.coco.loadImgs(img_id)[0]['file_name']

            # get distarctor
            dist_id = self.ids[distractor_idx]
            dist_img_id = self.coco.anns[dist_id]['image_id']
            distractor_path = self.coco.loadImgs(dist_img_id)[0]['file_name']

            # Convert image to tensor and pre-process using transform
            target_image = Image.open(os.path.join(self.image_dir, target_path)).convert('RGB')
            target_image = self.transform(target_image)

            distractor_image = Image.open(os.path.join(self.image_dir, distractor_path)).convert('RGB')
            distractor_image = self.transform(distractor_image)

            tokenizer = get_tokenizer("basic_english")
            tokens = tokenizer(str(target_caption).lower())
            # Convert caption to tensor of word ids, append start and end tokens.
            target_caption = []
            target_caption.append(self.vocab(self.vocab.start_word))

            # check if the sequence needs to be padded or truncated
            if self.max_sequence_length != 0:
                tokens = tokens[:self.max_sequence_length]

            target_caption.extend([self.vocab(token) for token in tokens])
            target_caption.append(self.vocab(self.vocab.end_word))
            target_caption = torch.Tensor(target_caption).long()
            return target_image, distractor_image, target_caption

        # obtain image if in test mode
        else:
            path = self.paths[idx]

            # Convert image to tensor and pre-process using transform
            PIL_image = Image.open(os.path.join(self.image_dir, path)).convert('RGB')
            orig_image = np.array(PIL_image)
            image = self.transform(PIL_image)
            # return original image and pre-processed image tensor
            return orig_image, image
        

    def get_func_train_indices(self):
        """
        Simple POC function returning two lists on indices for the functional training. 
        Returns a list of inidces for targets and a list f indices for distractors. 
        Captions are of same lengths for targets and distractors (will be optimized).
        
        Returns:
        -------
            list: (int, int)
                List of tuples of target and distractor indices, each for a single reference game iteration.
        """
        
        sel_length_t = np.random.choice(self.caption_lengths)

        all_indices_t = np.where([self.caption_lengths[i] == sel_length_t for i in np.arange(len(self.caption_lengths))])[0]

        indices_t = list(np.random.choice(all_indices_t, size=self.batch_size))
        possible_inds_dist = [x for x in np.arange(len(self.caption_lengths)) if x not in indices_t]
        indices_d = list(np.random.choice(possible_inds_dist, size=self.batch_size))
        
        return list(zip(indices_t, indices_d))

def _sort_images_by_category(download_dir, filename, is_train):
    """
    Sort image filenames into a dictionary with supercategory IDs as keys.

    Arguments:
    --------
        filename: str (instances file)
            Path to file containing image filenames and category annotations
    Returns:
    ------
        categories: dict
            Dict of form {category_id : [image file names] }
    """
    # check if json file with this info laready exists, otherwise export
    if os.path.exists("../../data/categories_to_image_paths.json"):
        print("Category to image IDs json file already exists!")
        # read existing file
        with open("../../data/categories_to_image_paths.json", "r", encoding="utf-8") as fp:
            categories = json.load(fp)
        return categories
    else:    
        # load instances file
        if is_train:
            path = os.path.join(download_dir, "annotations", "".join([filename, "_train2014.json"]))
        else: 
            path = os.path.join(download_dir, "annotations", "".join([filename, "_val2014.json"]))  

        # Load the json file.
        with open(path, "r", encoding="utf-8") as file:
            categories_raw = json.load(file)
            print("Loading categories for ", path)
    
        # load annotations section
        # TODO check how to map IDs to names elsewhere
        # category_names_ids = [(cat["id"], cat["name"]) for cat in categories_raw["categories"]]
        category_ids = [cat["id"] for cat in categories_raw["categories"]]
        print("List of category IDs: ", category_ids)
        # create empty dict for records
        categories = dict()
        # create dictinary with all the keys
        for i in category_ids:
            categories[i] = list()
        # build helper image dictionary with image IDs as keys
        images_raw = categories_raw["images"]
        image_dict = dict()
        for image in images_raw:
            image_dict[image["id"]] = image 

        # assign image filenames to the category keys
        # get image IDs from annotation entries
        annotation_ids = categories_raw["annotations"] 
        print("Number of records: ", len(annotation_ids))  
        for ann in annotation_ids:
            # get the image id of the entry
            image = ann["image_id"]
            # get the category id of the entry
            category = ann["category_id"]
            image_path = image_dict[image]["file_name"]
            # append imagepath to respective category
            categories[category].append(image_path)
            
        print("Dumping json ...")

        with open("../../data/categories_to_image_paths.json", "w") as fp:
            json.dump(categories, fp)

        return categories    

def _get_image_categories(download_dir, is_train):
    """
    Function mapping each image ID to all category annotations.
    """
    # check is file already exists
    if os.path.exists("../../data/imageIDs_to_categories.json"):
        print("Image IDs to category IDs json file already exists!")
        # read existing file
        with open("../../data/imageIDs_to_categories.json", "r", encoding="utf-8") as fp:
            images = json.load(fp)
        return images
    else:  
        if is_train: 
            file_path = os.path.join(download_dir, "annotations", "instances_train2014.json")
        else:
            file_path = os.path.join(download_dir, "annotations", "instances_val2014.json")    
        # load annotations file
        with open(file_path) as f:
            categories = json.load(f)
        # load categories to supercategories mapping file
        with open("../../data/categories_to_supercategories.json", "r") as cats:
            supercats = json.load(cats)    
        # create dictionary with image ids as keys
        images = dict()
        images_raw = categories["images"]
        for image in images_raw:
            images[image["id"]] = dict()
            images[image["id"]]["categories"] = list()
            images[image["id"]]["supercategories"] = list()
            
        for ann in categories["annotations"]:
            # check if category already included
            if ann["category_id"] in images[ann["image_id"]]["categories"]:
                pass
            else:
                images[ann["image_id"]]["categories"].append(ann["category_id"])
            # add supercategories
            supercat = supercats[str(ann["category_id"])]
            if supercat in images[ann["image_id"]]["supercategories"]:
                pass
            else:
                images[ann["image_id"]]["supercategories"].append(supercat)
            # check if the image is too crowded to be used in the within-category training
            if len(images[ann["image_id"]]["categories"]) > 6:
                images[ann["image_id"]]["has_many_categories"] = True 
            else:
                images[ann["image_id"]]["has_many_categories"] = False 

        # write out json
        with open("../../data/imageIDs_to_categories.json", "w") as fp:
            json.dump(images, fp)
        return images
        