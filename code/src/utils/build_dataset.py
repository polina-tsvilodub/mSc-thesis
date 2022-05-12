import json
import os
import torch
from . import dataset_utils 



def make_dataset(download_dir, filename, is_train):
    """
    Build torch dataset with images and captions.
    Images are first sorted into categories, images from 40 categories are chosen,
    then these are shuffled. 

    Arguments:
    --------
        download_dir: str
            Download directory where images and captions were donwloaded to
        filename: str
            File name prefix for reading captions json
        is_train: bool
            Flag is the split is train or val for constructing appropriate annotations file name     
    Returns:
    -----
        dataset: tfds
            Batched and suffled tfds instance of image + single caption pairs
    """

    print("Sorting images...")
    # _sort_images_by_category(
    #     download_dir=download_dir,
    #     filename="instances",
    #     is_train=is_train,
    # )
    print("Assign categories to images...")
    # _get_image_categories(
    #     download_dir=download_dir,
    #     is_train=is_train,
    # )
    # load the records
    # instantiate dataset here now 

    # instantiate get_loader 

    # _, filenames, captions = load_records.load_captions_data(
    #         download_dir=download_dir, 
    #         filename=filename, # annotations filename
    #         is_train=is_train,
        # )
    # load images   
    # not sure if this works  
    # print("Building image tfds") 
    # img_dataset = tf.data.Dataset.from_tensor_slices(filenames).map(
    #     preprocess_imgaes.load_image, num_parallel_calls=AUTOTUNE
    # )
    # # TODO
    # # don't forget to sample 1 caption out of 5 
    # # cap_dataset = tf.data.Dataset.from_tensor_slices(captions).map(
    # #     TODO, num_parallel_calls=AUTOTUNE
    # # )

    # dataset = tf.data.Dataset.zip((img_dataset, cap_dataset))
    # dataset = dataset.batch(BATCH_SIZE).shuffle(256).prefetch(AUTOTUNE)
    # return dataset


# Pass the list of images and the list of corresponding captions
# train_dataset = make_dataset(list(train_data.keys()), list(train_data.values()))

def get_loader(transform,
               mode='val',
               batch_size=1,
               vocab_threshold=None,
               vocab_file='../../data/vocab.pkl',
               start_word="START",
               end_word="END",
               unk_word="UNK",
               pad_word="PAD",
               vocab_from_file=True,
               num_workers=0,
               download_dir="../../../data/val/",
               vocab_from_pretrained=False,
               categorize_imgs=False,
              ):
    """Returns the data loader.
    Args:
      transform: Image transform.
      mode: One of 'train' or 'test'.
      batch_size: Batch size (if in testing mode, must have batch_size=1).
      vocab_threshold: Minimum word count threshold.
      vocab_file: File containing the vocabulary. 
      start_word: Special word denoting sentence start.
      end_word: Special word denoting sentence end.
      unk_word: Special word denoting unknown words.
      vocab_from_file: If False, create vocab from scratch & override any existing vocab_file.
                       If True, load vocab from from existing vocab_file, if it exists.
      num_workers: Number of subprocesses to use for data loading 
      cocoapi_loc: The location of the folder containing the COCO API: https://github.com/cocodataset/cocoapi
    """
    
    assert mode in ['train', 'test', 'val'], "mode must be one of 'train' or 'test'."
    
    if vocab_from_file==False: assert mode=='train' or mode=='val', "To generate vocab from captions file, must be in training mode (mode='train')."

    # Based on mode (train, val, test), obtain img_folder and annotations_file.
    if mode == 'val':
        if vocab_from_file==True: assert os.path.exists(vocab_file), "vocab_file does not exist.  Change vocab_from_file to False to create vocab_file."
        img_folder = os.path.join(download_dir, "val2014/") 
        annotations_file = os.path.join(download_dir, 'annotations/captions_val2014.json')
    if mode == 'train':
        if vocab_from_file==True: assert os.path.exists(vocab_file), "vocab_file does not exist.  Change vocab_from_file to False to create vocab_file."
        img_folder = os.path.join(download_dir, "train2014/") 
        annotations_file = os.path.join(download_dir, 'annotations/captions_train2014.json')
    if mode == 'test':
        assert batch_size==1, "Please change batch_size to 1 if testing your model."
        assert os.path.exists(vocab_file), "Must first generate vocab.pkl from training data."
        assert vocab_from_file==True, "Change vocab_from_file to True."
        img_folder = os.path.join(download_dir, "val2014/") #'test2014/'
        annotations_file = os.path.join(download_dir, 'annotations/captions_val2014.json') #image_info_test2014

    # TODO: alternatively, modify / sort the image IDS on the instantiated dataset.ids object 
    # COCO caption dataset.
    dataset = dataset_utils.COCOCaptionsDataset(
        file=annotations_file,
        download_dir = download_dir, 
        img_transform=transform,
        text_transform=None,
        batch_size=batch_size,
        mode=mode,
        vocab_threshold=vocab_threshold,
        vocab_file=vocab_file,
        start_token=start_word,
        end_token=end_word,
        unk_token=unk_word,
        pad_token=pad_word,
        vocab_from_file=vocab_from_file,
        vocab_from_pretrained=vocab_from_pretrained,
        max_sequence_length=15,
        categorize_imgs=categorize_imgs,
    )
    

    if mode == 'train':
        # Randomly sample a caption length, and sample indices with that length.
        indices = dataset.get_func_train_indices()
        # Create and assign a batch sampler to retrieve a batch with the sampled indices.
        initial_sampler = torch.utils.data.sampler.SubsetRandomSampler(indices=indices)
        # data loader for COCO dataset.
        data_loader = torch.utils.data.DataLoader(
            dataset=dataset, 
            num_workers=num_workers,
            batch_sampler=torch.utils.data.sampler.BatchSampler(sampler=initial_sampler,
                                                                batch_size=dataset.batch_size,
                                                                drop_last=False))
    elif mode == 'val':
        # Randomly sample a caption length, and sample indices with that length.
        indices = dataset.get_func_train_indices()
        # Create and assign a batch sampler to retrieve a batch with the sampled indices.
        initial_sampler = torch.utils.data.sampler.SubsetRandomSampler(indices=indices)
        # data loader for COCO dataset.
        data_loader = torch.utils.data.DataLoader(
            dataset=dataset, 
            num_workers=num_workers,
            batch_sampler=torch.utils.data.sampler.BatchSampler(sampler=initial_sampler,
                                                                batch_size=dataset.batch_size,
                                                                drop_last=False))
    else:
        data_loader = torch.utils.data.DataLoader(
            dataset=dataset,               
            batch_size=dataset.batch_size,
            shuffle=True,
            num_workers=num_workers,
        )

    return data_loader