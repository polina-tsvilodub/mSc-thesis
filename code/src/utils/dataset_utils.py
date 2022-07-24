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
            pairs="random",
            vocab_from_pretrained=False,
            max_sequence_length=0,
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
        self.pairs = pairs

        # read categories2img
        with open("../../../data/categories_to_image_IDs_train_filtered.json", "r") as fp:
            f = json.load(fp)
        self.categories2image = f    
        self.category_ids = list(f.keys())
        # read img2categories
        with open("../../../data/imageIDs_to_categories_train_filtered.json", "r") as fp:
            f = json.load(fp)
        self.imgs2cats = f
        # read imgID to annIDs mapping file
        with open("imgID2annID.json", "r") as fp:
            f = json.load(fp)
        self.imgID2annID = f
        
        # some distinctions below for Train and test mode
        if mode == "train":
            self.image_dir = os.path.join(download_dir, "train2014") # download_dir needs to be data/train/ then 
            self.coco = COCO(file) # os.path.join(download_dir, file)
            # _ids = list(self.coco.anns.keys())
            # shuffle(_ids)
            ####
            # read imd2annID file. select N images, get all ann IDs => ids
            with open("imgID2annID.json", "r") as fp:
                f = json.load(fp)
            if pairs == "similar":
                imgIDs4train = torch.load("../../../data/ref_game_similar_img_IDs_filtered.pt")[:30000] 
            else:
                imgIDs4train = list(f.keys())[:30000] # 
            _ids = [(f[i], i) for i in imgIDs4train] # list of tuples of shape (annID_lst, imgID)
            shuffle(_ids)
            _ann_ids_flat = [i for lst in _ids for i in lst[0]]
            
            ####
            # retrieve a subset of images for pretraining
            if dataset_path is not None:
                if num_imgs != 0:
                    self.ids = torch.load(dataset_path).tolist()[:num_imgs]
                else:
                    self.ids = torch.load(dataset_path).tolist()
            else:
                self.ids = torch.load("../train_logs/pretrain_img_IDs_2imgs_512dim.pt").tolist() #_ann_ids_flat #torch.load("pretrain_img_IDs_2imgs_512dim_100000imgs.pt").tolist()#_ids[:70000] list(self.coco.anns.keys()) #
            
            if dataset_path == "./val_split_IDs_from_COCO_train_tensor.pt":
                self._img_ids_flat = [self.coco.loadAnns(i)[0]['image_id'] for i in self.ids]
            else:   
                self._img_ids_flat = [i[1] for i in _ids for x in i[0]]
            

            # set the image IDs for validation during early stopping to avoid overlapping images
            self.ids_val = torch.load("../train_logs/pretrain_val_img_IDs_2imgs.pt").tolist() #_ids[70000:73700]
            print('Obtaining caption lengths...')
            tokenizer = get_tokenizer("basic_english") # nltk.tokenize.word_tokenize(str(self.coco.anns[self.ids[index]]['caption']).lower())
            all_tokens = [tokenizer(str(self.coco.anns[self.ids[index]]['caption']).lower()) for index in tqdm(np.arange(len(self.ids)))] 
            all_tokens_val = [tokenizer(str(self.coco.anns[self.ids_val[index]]['caption']).lower()) for index in tqdm(np.arange(len(self.ids_val)))] 
            self.caption_lengths = [len(token) for token in all_tokens]
            self.caption_lengths_val = [len(token) for token in all_tokens_val] 
            # print pretraining IDs for later separation from functional training
            # save used indices to torch file
            # torch.save(torch.tensor(self.ids), "train_logs/ref-game_img_IDs_15000_coco_pretrainGrid_decr05.pt")
            # torch.save(torch.tensor(self.ids), "train_logs/15000_coco_hyperparams_search_Lf_sampling_tracked.pt")
            
            # torch.save(torch.tensor(self.ids), "train_logs/final/pretrain_img_IDs_teacher_forcing_desc05_pureDecoding_vocab4000_padding_cont_4.pt")

        elif mode == "val":
            with open("./imgID2annID_val.json", "r") as fp:
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
        

    def get_func_train_indices(self, i_step):
        """
        Simple POC function returning two lists on indices for the functional training. 
        Returns a list of inidces for targets and a list f indices for distractors. 
        Caption lengths are the same for targets by sampling design, distractor captions are padded upon batch construction.

        Returns:
        -------
            list: (int, int)
                List of tuples of target and distractor indices, each for a single reference game iteration.
        """
        
        # sel_length_t = np.random.choice(self.caption_lengths)

        # all_indices_t = np.where([self.caption_lengths[i] == sel_length_t for i in np.arange(len(self.caption_lengths))])[0]

        # indices_t = list(np.random.choice(all_indices_t, size=self.batch_size))
        # indices_t = list(np.random.choice(np.arange(len(self.ids)), size=self.batch_size))
        indices_t = list(range((i_step-1)*self.batch_size, i_step*self.batch_size))
        # print("Indices_t ", indices_t)
        # retrieve image ids of sampled ids to make sure we don't get target distractor pairs
        # consisiting of same images
        imgIDs_t = [self._img_ids_flat[i] for i in indices_t]
        possible_inds_dist = [x for x in np.arange(len(self.caption_lengths)) if x not in indices_t and self._img_ids_flat[x] not in imgIDs_t]
        indices_d = list(np.random.choice(possible_inds_dist, size=self.batch_size, replace=False))
        
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

        # get batch indices
        indices_t = list(range((i_step-1)*self.batch_size, i_step*self.batch_size))
        # for each image, get its categories and iteratively choose a suitable distractor
        targets_categories = [self.imgs2cats[str(self._img_ids_flat[i])] for i in indices_t]
        distractor_inds = []
        distractor_ann_inds = []
        
        for num, c in enumerate(targets_categories):
            print("Target categories: ", c)
            # align the target and distractor categories 
            possible_inds = []
            # get distractor images matching in at least three of the target categories
            for i, c_t in enumerate(c["categories"][:3]):
                print("I, c_t ", i, c_t)
                if i == 0:
                    possible_inds = list(set.intersection(set(self.categories2image[str(c_t)]), set(self._img_ids_flat) - set([self._img_ids_flat[i] for i in indices_t]))) # remove used target img ids here to avoid having targets and dists from same images
                else:
                    possible_inds = list(set.intersection(set(self.categories2image[str(c_t)]), set(possible_inds)))
            
            print("num of possible inds ", len(possible_inds))
            dist_ind = possible_inds[0]
            print("Dist ind ", dist_ind)
            j = 1
            while dist_ind == indices_t[num]:
                dist_ind = possible_inds[j]
                j += 1 # TODO check it does not get out of range
            distractor_inds.append(dist_ind)   
            print("Dist ind ", dist_ind)
            # convert distractor image id to index of the corresponding annotation ID
            distractor_ann_ind = np.where([self._img_ids_flat[i] == dist_ind for i in range(len(self._img_ids_flat))])[0]
            print("distractor ann ind ", distractor_ann_ind)
            distractor_ann_inds.append(distractor_ann_ind[0])
        print("Distractor inds final ", distractor_inds)
        print("Distractor ann inds final ", distractor_ann_inds)
    
        inds_tuples = list(zip(indices_t, distractor_ann_inds))
        print("inds_tuples ", inds_tuples)
        return inds_tuples


    def collate_distractors(self, batch):
        
        target_image, distractor_image, target_features, distractor_features, target_caption, distractor_caps = [], [], [], [], [], []
    
        for (t_i, d_i, t_f, d_f, targ_c, dist) in batch:
            target_image.append(t_i)
            distractor_image.append(d_i)
            target_features.append(t_f)
            distractor_features.append(d_f)
            target_caption.append(targ_c)
            distractor_caps.append(dist)
        
        targets_list = pad_sequence(target_caption, batch_first=True, padding_value=self.vocab(self.vocab.pad_word))
        text_list = pad_sequence(distractor_caps, batch_first=True, padding_value=self.vocab(self.vocab.pad_word))
        
        target_image = torch.stack(target_image)
        distractor_image = torch.stack(distractor_image)
        target_features = torch.stack(target_features)
        distractor_features = torch.stack(distractor_features)
        
        return target_image, distractor_image, target_features, distractor_features, targets_list, text_list

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
        dataset_path,
        num_imgs,
        pairs,
        vocab_from_pretrained=False,
        max_sequence_length=0,
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
        # categories with respective possible values, for similar pairs expt
        self.categories = {
            'floor_hue': [0,0.1,0.2,0.30000000000000004,0.4,0.5,0.6000000000000001,0.7000000000000001,0.8,0.9],
            'wall_hue': [0,0.1,0.2,0.30000000000000004,0.4,0.5,0.6000000000000001,0.7000000000000001,0.8,0.9], 
            'object_hue': [0,0.1,0.2,0.30000000000000004,0.4,0.5,0.6000000000000001,0.7000000000000001,0.8,0.9],
            'scale': [0.75,0.8214285714285714,0.8928571428571428,0.9642857142857143,1.0357142857142856,1.1071428571428572,1.1785714285714286,1.25], 
            'shape': [0,1,2,3], 
            'orientation': [-30,-25.714285714285715,-21.42857142857143,-17.142857142857142,-12.857142857142858,-8.571428571428573,-4.285714285714285,0, 4.285714285714285,8.57142857142857,12.857142857142854,17.14285714285714,21.42857142857143,25.714285714285715,30],
        }
        with open("./categories2imgIDs_3dshapes_fixed_float_train.json", "r") as fp:
            self.cats2imgIDs = json.load(fp)

        if mode == "train":
            self.image_dir = os.path.join(download_dir, "3dshapes_np.npy") # download_dir needs to be data/train/ then 
            self.images = np.load(self.image_dir) # ["images"] # os.path.join(download_dir, file)
            self.numeric_labels = np.load(
                os.path.join(download_dir, "3dshapes_labels_np.npy")
            )
            with open("../../../data/3dshapes_captions_fixed.json", "r") as fp:
                self.labels = json.load(fp)
            with open("../../../data/3dshapes_captions_short.json", "r") as fp:
                self.labels_short = json.load(fp)
                
            
            if dataset_path is not None:
                if num_imgs != 0:
                    imgIDs4train = torch.load(dataset_path)[:num_imgs]
                else:
                    imgIDs4train = torch.load(dataset_path)
            else:
                imgIDs4train = np.random.choice(list(self.labels.keys()), 30000) # train split, torch.load("pretrain_img_IDs_unique_3dshapes_final_pretrain.pt") #
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
            all_tokens = [tokenizer(str(self.ids[index]).replace("-", " ")) for index in tqdm(np.arange(len(self.ids)))] 
            self.caption_lengths = [len(token) for token in all_tokens]
            
            # print pretraining IDs for later separation from functional training
            # save used indices to torch file
            # print("Type check before saving: ", type(imgIDs4train[0]))
            # torch.save(imgIDs4train, "ref_game_img_IDs_unique_imgIDs4train_3dshapes.pt")
            # torch.save(self._img_ids_flat, "pretrain_img_IDs_flat_3dshapes_short.pt")
            # torch.save(self.ids, "train_logs/final/pretrain_img_IDs_3dshapes_exh_teacher_forcing_desc05_pureDecoding_vocab49_padding_cont4.pt")
            
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
            
            tokens = self.tokenizer(str(target_lbl).lower().replace("-", " "))
            tokens_dist = self.tokenizer(str(dist_lbl).lower().replace("-", " "))
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

    def get_func_train_indices(self, i_step):
        """
        Simple POC function returning two lists on indices for the functional training. 
        Returns a list of inidces for targets and a list f indices for distractors. 
        Captions are of same lengths for targets and distractors (will be optimized).
        
        Returns:
        -------
            list: (int, int)
                List of tuples of target and distractor indices, each for a single reference game iteration.
        """
        indices_t = list(range((i_step-1)*self.batch_size, i_step*self.batch_size))
        # print("Indices_t ", indices_t)
        # retrieve image ids of sampled ids to make sure we don't get target distractor pairs
        # consisiting of same images
        imgIDs_t = [self._img_ids_flat[i] for i in indices_t]
        possible_inds_dist = [x for x in np.arange(len(self.caption_lengths)) if x not in indices_t and self._img_ids_flat[x] not in imgIDs_t]
        indices_d = list(np.random.choice(possible_inds_dist, size=self.batch_size, replace=False))
        
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
        
        targets_list = pad_sequence(target_caption, batch_first=True, padding_value=self.vocab(self.vocab.pad_word))
        text_list = pad_sequence(distractor_caps, batch_first=True, padding_value=self.vocab(self.vocab.pad_word))
        
        target_image = torch.stack(target_image)
        distractor_image = torch.stack(distractor_image)
        target_features = torch.stack(target_features)
        distractor_features = torch.stack(distractor_features)
        
        return target_image, distractor_image, target_features, distractor_features, targets_list, text_list

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

            # get ann / img ID slice
            target_img_ids = self._img_ids_flat[(i_step-1)*self.batch_size : i_step*self.batch_size]
            target_labels = [self.numeric_labels[int(i)] for i in target_img_ids]
            
            # select at random 3 categories along which the image should be constant in this batch
            sel_categories = np.random.choice(list(self.categories.keys()), size=3, replace=False)
            sel_categories_inds = [list(self.categories.keys()).index(c) for c in sel_categories]
            print("Selected categories: ", sel_categories)
            # create a container for the distractor indices
            distractor_inds = []
            
            # select the indices where the three categories + values have desired values
            
            # iterate over each sample to find the right distractor
            for lbl in target_labels:
                print("Lbl ", lbl)
                # create a placeholder for intersecting the indices for that 
                possible_inds = []
                # get values of sampled fixed categories and matching distractor indices
                for i, c_i in enumerate(list(zip(sel_categories, sel_categories_inds))):
                    print("Ind retrieval loop for lbl: ", i, c_i)
                    if i == 0:
                        possible_inds = self.cats2imgIDs[c_i[0]][str(lbl[c_i[1]])]
                    else:
                        possible_inds = list(set.intersection(set(self.cats2imgIDs[c_i[0]][str(lbl[c_i[1]])]), set(possible_inds)))
                # get one distractor 
                distractor_img_ind = possible_inds[0]
                print("distractor_img_ind", distractor_img_ind)
                # convert retrieved distractor image index into distractor annotation index
                distractor_ann_ind = np.where([self._img_ids_flat[i] == str(distractor_img_ind) for i in range(len(self._img_ids_flat))])[0]
                print("distractor_ann_id", distractor_ann_ind)
                distractor_inds.append(int(distractor_ann_ind[0]))
            
                            
            print("Selected cats: ", sel_categories)
            print("Number of distractor inds: ", len(distractor_inds))
            
            print("target img ind: ", target_img_ids)
            inds_tuples = list(zip(list(range((i_step-1)*self.batch_size, i_step*self.batch_size)), distractor_inds)) #list(zip(indices_t, indices_d))
            print("tuples: ", inds_tuples)
            # double check that target and distractor aren't the same image
            
            return inds_tuples