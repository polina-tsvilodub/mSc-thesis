import json
import os
import tensorflow as tf
from . import preprocess_images
from . import load_records 

# def process_input(img_path, captions):
#     return decode_and_resize(img_path), vectorization(captions)

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

def make_dataset(download_dir, filename, is_train):
    """
    Build tensorflow dataset with images and captions.
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
    _sort_images_by_category(
        download_dir=download_dir,
        filename="instances",
        is_train=is_train,
    )
    print("Assign categories to images...")
    _get_image_categories(
        download_dir=download_dir,
        is_train=is_train,
    )
    # load the records
    _, filenames, captions = load_records.load_captions_data(
            download_dir=download_dir, 
            filename=filename, # annotations filename
            is_train=is_train,
        )
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
