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
import h5py
from torch.nn.utils.rnn import pad_sequence
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
            embedded_imgs,
            dataset_path,
            num_imgs,
            pairs,
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
        self.embedded_imgs=embedded_imgs
        self.tokenizer = get_tokenizer("basic_english")

        # read categories2img
        # with open("../../../data/categories_to_image_IDs_train_filtered.json", "r") as fp:
        #     f = json.load(fp)
        # self.categories2image = f    
        # self.category_ids = list(f.keys())
        # # read img2categories
        # with open("../../../data/imageIDs_to_categories_train_filtered.json", "r") as fp:
        #     f = json.load(fp)
        # self.imgs2cats = f
        # # read imgID to annIDs mapping file
        # with open("imgID2annID.json", "r") as fp:
        #     f = json.load(fp)
        # self.imgID2annID = f
        
        # some distinctions below for Train and test mode
        if mode == "train":
            self.image_dir = os.path.join(download_dir, "train2014") # download_dir needs to be data/train/ then 
            self.coco = COCO(file) # os.path.join(download_dir, file)
            # _ids = list(self.coco.anns.keys())
            # shuffle(_ids)
            ####
            # read imd2annID file. select N images, get all ann IDs => ids
            with open("notebooks/imgID2annID.json", "r") as fp:
                f = json.load(fp)
            imgIDs4train = list(f.keys())[:30000] # 
            _ids = [(f[i], i) for i in imgIDs4train] # list of tuples of shape (annID_lst, imgID)
            shuffle(_ids)
            _ann_ids_flat = [i for lst in _ids for i in lst[0]]
            self._img_ids_flat = [i[1] for i in _ids for x in i[0]]
            
            ####
            # retrieve a subset of images for pretraining
            if dataset_path is not None:
                if num_imgs != 0:
                    self.ids = torch.load(dataset_path).tolist()[:num_imgs]
                else:
                    self.ids = torch.load(dataset_path).tolist()
            else:
                self.ids = _ann_ids_flat #torch.load("pretrain_img_IDs_2imgs_512dim_100000imgs.pt").tolist()#_ids[:70000] list(self.coco.anns.keys()) #
            # set the image IDs for validation during early stopping to avoid overlapping images
            self.ids_val = torch.load("train_logs/pretrain_val_img_IDs_2imgs.pt").tolist() #_ids[70000:73700]
            print('Obtaining caption lengths...')
            tokenizer = get_tokenizer("basic_english") # nltk.tokenize.word_tokenize(str(self.coco.anns[self.ids[index]]['caption']).lower())
            all_tokens = [tokenizer(str(self.coco.anns[self.ids[index]]['caption']).lower()) for index in tqdm(np.arange(len(self.ids)))] 
            all_tokens_val = [tokenizer(str(self.coco.anns[self.ids_val[index]]['caption']).lower()) for index in tqdm(np.arange(len(self.ids_val)))] 
            self.caption_lengths = [len(token) for token in all_tokens]
            self.caption_lengths_val = [len(token) for token in all_tokens_val] 
            # print pretraining IDs for later separation from functional training
            # save used indices to torch file
            # torch.save(torch.tensor(self.ids), "train_logs/ref-game_img_IDs_15000_coco_lf01.pt")
            torch.save(torch.tensor(self.ids), "train_logs/pretrain_img_IDs_30000_coco_teacher_forcing_scheduled_desc_05_byEp.pt")
            
            # torch.save(torch.tensor(self.ids_val), "pretrain_val_img_IDs_2imgs_1024dim.pt")

        elif mode == "val":
            with open("notebooks/imgID2annID_val.json", "r") as fp:
                f = json.load(fp)
            imgIDs4val = list(f.keys())
            _ids = [(f[i], i) for i in imgIDs4val] # list of tuples of shape (annID_lst, imgID)
            shuffle(_ids)
            _ann_ids_flat = [i for lst in _ids for i in lst[0]]
            self._img_ids_flat = [i[1] for i in _ids for x in i[0]]

            self.image_dir = os.path.join(download_dir, "val2014")
            self.coco = COCO(file) # os.path.join(download_dir, file)
            self.ids = _ann_ids_flat #list(self.coco.anns.keys())#[:1000]
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
            distractor_caption = self.coco.anns[dist_id]['caption']
            dist_img_id = self.coco.anns[dist_id]['image_id']
            distractor_path = self.coco.loadImgs(dist_img_id)[0]['file_name']

            # Convert image to tensor and pre-process using transform
            target_image = Image.open(os.path.join(self.image_dir, target_path)).convert('RGB')
            target_image = self.transform(target_image)
            # Get the pre-extracted features
            target_features = self.embedded_imgs[str(ann_id)]#[target_idx, :].squeeze(0) #torch.index_select(self.embedded_imgs, 0, target_idx).squeeze(1)

            distractor_image = Image.open(os.path.join(self.image_dir, distractor_path)).convert('RGB')
            distractor_image = self.transform(distractor_image)
            # Get pre-extracted features
            distractor_features = self.embedded_imgs[str(dist_id)]#[distractor_idx, :].squeeze(0) # torch.index_select(self.embedded_imgs, 0, distractor_idx).squeeze(1)

            tokens = self.tokenizer(str(target_caption).lower())
            tokens_dist = self.tokenizer(str(distractor_caption).lower())
            # Convert caption to tensor of word ids, append start and end tokens.
            target_caption = []
            distractor_caption = []
            target_caption.append(self.vocab(self.vocab.start_word))
            distractor_caption.append(self.vocab(self.vocab.start_word))

            # check if the sequence needs to be padded or truncated
            if self.max_sequence_length != 0:
                tokens = tokens[:self.max_sequence_length]
                tokens_dist = tokens_dist[:self.max_sequence_length]

            target_caption.extend([self.vocab(token) for token in tokens])
            target_caption.append(self.vocab(self.vocab.end_word))
            target_caption = torch.Tensor(target_caption).long()

            distractor_caption.extend([self.vocab(token) for token in tokens_dist])
            distractor_caption.append(self.vocab(self.vocab.end_word))
            distractor_caption = torch.Tensor(distractor_caption).long()

            return target_image, distractor_image, target_features, distractor_features, target_caption, distractor_caption

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
        Caption lengths are the same for targets by sampling design, distractor captions are padded upon batch construction.

        Returns:
        -------
            list: (int, int)
                List of tuples of target and distractor indices, each for a single reference game iteration.
        """
        
        sel_length_t = np.random.choice(self.caption_lengths)

        all_indices_t = np.where([self.caption_lengths[i] == sel_length_t for i in np.arange(len(self.caption_lengths))])[0]

        indices_t = list(np.random.choice(all_indices_t, size=self.batch_size))
        # retrieve image ids of sampled ids to make sure we don't get target distractor pairs
        # consisiting of same images
        imgIDs_t = [self._img_ids_flat[i] for i in indices_t]
        possible_inds_dist = [x for x in np.arange(len(self.caption_lengths)) if x not in indices_t and self._img_ids_flat[x] not in imgIDs_t]
        indices_d = list(np.random.choice(possible_inds_dist, size=self.batch_size))
        
        return list(zip(indices_t, indices_d))

    def get_func_similar_train_indices(self):
        """
        Simple POC function returning two lists on indices for the functional training. 
        Returns a list of inidces for targets and a list f indices for distractors. 
        Captions are of same lengths for targets and distractors (will be optimized).
        
        Returns:
        -------
            list: (int, int)
                List of tuples of target and distractor indices, each for a single reference game iteration.
        """

        # TODO checks from construction of pairs w all captions per iamge need to be added
      
        # select at random a category of interest
        # the idea is to have by-batch catrgorization
        sel_category = np.random.choice(self.category_ids)
        all_indices_cat = self.categories2image[sel_category]
        # sample a batch, create tuples
        indices_t = list(np.random.choice(all_indices_cat, size=self.batch_size))
        possible_inds_dist = [x for x in all_indices_cat if x not in indices_t]
        indices_d = list(np.random.choice(possible_inds_dist, size=self.batch_size))
        inds_tuples = list(zip(indices_t, indices_d))
        # check that for each pair, the categories overlap between target and distractor matches my criteria
        for tup in inds_tuples:
            common_cats = list(set.intersection(set(self.imgs2cats[str(tup[0])]['categories']), set(self.imgs2cats[str(tup[1])]['categories'])))
            print("common cats before while: ", common_cats)
            print("Len common cats: ", len(common_cats))
            print("target cats: ", set(self.imgs2cats[str(tup[0])]['categories']))
            print("Len ", len(set(self.imgs2cats[str(tup[0])]['categories'])))
            print("dist cats: ", set(self.imgs2cats[str(tup[1])]['categories']))
            while len(common_cats) < 3 and len(common_cats) != len(self.imgs2cats[str(tup[0])]['categories']):
                print("common cats in while: ", common_cats)
                tup = tuple([tup[0], np.random.choice(possible_inds_dist, size=1).item()])
                common_cats = list(set.intersection(set(self.imgs2cats[str(tup[0])]['categories']), set(self.imgs2cats[str(tup[1])]['categories'])))
                # sample a new target dist pair
                
#         all_indices_t = np.where([self.caption_lengths[i] == sel_length_t for i in np.arange(len(self.caption_lengths))])[0]

#         indices_t = list(np.random.choice(all_indices_t, size=self.batch_size))
#         possible_inds_dist = [x for x in np.arange(len(self.caption_lengths)) if x not in indices_t]
#         indices_d = list(np.random.choice(possible_inds_dist, size=self.batch_size))
        
        return inds_tuples#list(zip(indices_t, indices_d))


    def collate_distractors(self, batch):
        
        target_image, distractor_image, target_features, distractor_features, target_caption, distractor_caps = [], [], [], [], [], []
    
        for (t_i, d_i, t_f, d_f, targ_c, dist) in batch:
            target_image.append(t_i)
            distractor_image.append(d_i)
            target_features.append(t_f)
            distractor_features.append(d_f)
            target_caption.append(targ_c)
            distractor_caps.append(dist)
        
        text_list = pad_sequence(distractor_caps, batch_first=True, padding_value=self.vocab(self.vocab.pad_word))
        
        target_image = torch.stack(target_image)
        distractor_image = torch.stack(distractor_image)
        target_features = torch.stack(target_features)
        distractor_features = torch.stack(distractor_features)
        target_caption = torch.stack(target_caption)

        return target_image, distractor_image, target_features, distractor_features, target_caption, text_list

class threeDshapes_Dataset(Dataset):
    """
    Dataset class for loading the dataset of images and captions from the 3dshapes dataset.
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
        embedded_imgs,
        vocab_from_pretrained=False,
        max_sequence_length=0,
        categorize_imgs=False,
        ):

        self.transform = img_transform
        self.mode = mode # train or test or val
        self.batch_size = batch_size
        self.vocab = vocabulary.Vocabulary(vocab_threshold, vocab_file, 
                                start_token, end_token, unk_token, 
                                file, pad_token, vocab_from_file, vocab_from_pretrained) 
        self.max_sequence_length = max_sequence_length
        self.pad_token=pad_token
        self.embedded_imgs=embedded_imgs
        self.tokenizer = get_tokenizer("basic_english")

        if mode == "train":
            self.image_dir = os.path.join(download_dir, "3dshapes_np.npy") # download_dir needs to be data/train/ then 
            self.images = np.load(self.image_dir) # ["images"] # os.path.join(download_dir, file)
            with open("../../data/3dshapes_captions_fixed.json", "r") as fp:
                self.labels = json.load(fp)
            with open("../../data/3dshapes_captions_short.json", "r") as fp:
                self.labels_short = json.load(fp)
                
            
            imgIDs4train = torch.load("ref_game_imgIDs_unique_wShortCaps_3dshapes.pt") #np.random.choice(list(self.labels.keys()), 30000) # train split, torch.load("pretrain_img_IDs_unique_3dshapes_final_pretrain.pt") #
            print("len of img ids: ", len(imgIDs4train))
            _ids = [(self.labels[i], i) for i in imgIDs4train] # list of tuples of shape (annID_lst, imgID)
            # _ids_short = [(self.labels_short[i], i) for i in imgIDs4train] 

            # _anns_flat = [i for lst in _ids for i in np.random.choice(lst[0], 3)] # only select 5 random captions among 40 possible
            # _anns_flat_short = [i for lst in _ids_short for i in np.random.choice(lst[0], 3)] # only select 5 random captions among 40 possible 
            # self._img_ids_flat = [i[1] for i in _ids for x in i[0][:6]]
            self._img_ids_flat = [i[1] for i in _ids for x in i[0][:5]]
            # _unique_IDs = list(set(self._img_ids_flat))
            _anns_flat = []
            for i in imgIDs4train:
                # short_caps = np.random.choice(self.labels_short[i], 3)
                long_caps = np.random.choice(self.labels[i], 5) # 3
                # _anns_flat.extend(short_caps)
                _anns_flat.extend(long_caps)
            self.ids = _anns_flat
            print("IDS ", len(self.ids), self.ids[:15])
            print("img IDs ", len(self._img_ids_flat), self._img_ids_flat[:15])

            print('Obtaining caption lengths...')
            tokenizer = get_tokenizer("basic_english") # nltk.tokenize.word_tokenize(str(self.coco.anns[self.ids[index]]['caption']).lower())
            all_tokens = [tokenizer(str(self.ids[index]).replace("-", "")) for index in tqdm(np.arange(len(self.ids)))] 
            self.caption_lengths = [len(token) for token in all_tokens]
            
            # print pretraining IDs for later separation from functional training
            # save used indices to torch file
            # print("Type check before saving: ", type(imgIDs4train[0]))
            # torch.save(imgIDs4train, "ref_game_img_IDs_unique_imgIDs4train_3dshapes.pt")
            # torch.save(self._img_ids_flat, "pretrain_img_IDs_flat_3dshapes_short.pt")
            # torch.save(self.ids, "pretrain_anns_flat_3dshapes_short.pt")
            
        if mode == "val":
            pass

    def __len__(self):
        return len(self.ids)
        # for extracting resnet features:
        # return len(self.images)

    def __getitem__(self, idx):
        if self.mode != "test":
            target_idx = idx[0]
            distractor_idx = idx[1]
            # just accessing images directly for resnet feature extraction
            target_img = self.images[int(self._img_ids_flat[target_idx])]
            target_lbl = self.ids[target_idx]
            
            dist_img = self.images[int(self._img_ids_flat[distractor_idx])] # self.images[int(distractor_idx)] #
            dist_lbl = self.ids[distractor_idx]
            target_img = np.asarray(target_img).astype('uint8')

            dist_img = np.asarray(dist_img).astype('uint8')
            target_img = self.transform(target_img)
            dist_img = self.transform(dist_img)
            
            target_features = self.embedded_imgs[int(self._img_ids_flat[target_idx])]#.squeeze(0)
            distractor_features = self.embedded_imgs[int(self._img_ids_flat[distractor_idx])]#.squeeze(0)
            
            tokens = self.tokenizer(str(target_lbl).lower().replace("-", ""))
            tokens_dist = self.tokenizer(str(dist_lbl).lower().replace("-", ""))
            # Convert caption to tensor of word ids, append start and end tokens.
            target_caption = []
            distractor_caption = []
            target_caption.append(self.vocab(self.vocab.start_word))
            distractor_caption.append(self.vocab(self.vocab.start_word))

            # check if the sequence needs to be padded or truncated
            if self.max_sequence_length != 0:
                tokens = tokens[:self.max_sequence_length]
                tokens_dist = tokens_dist[:self.max_sequence_length]

            target_caption.extend([self.vocab(token) for token in tokens])
            target_caption.append(self.vocab(self.vocab.end_word))
            target_caption = torch.Tensor(target_caption).long()

            distractor_caption.extend([self.vocab(token) for token in tokens_dist])
            distractor_caption.append(self.vocab(self.vocab.end_word))
            distractor_caption = torch.Tensor(distractor_caption).long()
            
            return target_img, dist_img, target_features, distractor_features, target_caption, distractor_caption

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
        # retrieve image ids of sampled ids to make sure we don't get target distractor pairs
        # consisiting of same images
        imgIDs_t = [self._img_ids_flat[i] for i in indices_t]
        possible_inds_dist = list(set(np.arange(len(self.caption_lengths)) ) - set(indices_t) ) #[x for x in np.arange(len(self.caption_lengths)) if x not in indices_t and self._img_ids_flat[x] not in imgIDs_t]
        checked_ind_d = set(possible_inds_dist) - set([self._img_ids_flat[x] for x in possible_inds_dist])
        # print("Len check ind ", len(checked_ind_d))
        indices_d = list(np.random.choice(list(checked_ind_d), size=self.batch_size))
        
        return list(zip(indices_t, indices_d))

    def collate_distractors(self, batch):
        
        target_image, distractor_image, target_features, distractor_features, target_caption, distractor_caps = [], [], [], [], [], []
    
        for (t_i, d_i, t_f, d_f, targ_c, dist) in batch:
            target_image.append(t_i)
            distractor_image.append(d_i)
            target_features.append(t_f)
            distractor_features.append(d_f)
            target_caption.append(targ_c)
            distractor_caps.append(dist)
        
        text_list = pad_sequence(distractor_caps, batch_first=True, padding_value=self.vocab(self.vocab.pad_word))
        
        target_image = torch.stack(target_image)
        distractor_image = torch.stack(distractor_image)
        target_features = torch.stack(target_features)
        distractor_features = torch.stack(distractor_features)
        target_caption = torch.stack(target_caption)

        return target_image, distractor_image, target_features, distractor_features, target_caption, text_list

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
        