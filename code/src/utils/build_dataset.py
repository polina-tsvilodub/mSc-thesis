import tensorflow as tf
from . import preprocess_images 
from . import load_records 

# def process_input(img_path, captions):
#     return decode_and_resize(img_path), vectorization(captions)


def make_dataset(images, captions):
    """
    
    """

    # load the records
    _, filenames, captions = load_captions_data(
            download_dir="data/", # TODO
            filename="data/", # TODO
        )
    # load images   
    # not sure if this works   
    img_dataset = tf.data.Dataset.from_tensor_slices(filenames).map(
        load_image, num_parallel_calls=AUTOTUNE
    )

    cap_dataset = tf.data.Dataset.from_tensor_slices(captions).map(
        TODO, num_parallel_calls=AUTOTUNE
    )

    dataset = tf.data.Dataset.zip((img_dataset, cap_dataset))
    dataset = dataset.batch(BATCH_SIZE).shuffle(256).prefetch(AUTOTUNE)
    return dataset


# Pass the list of images and the list of corresponding captions
# train_dataset = make_dataset(list(train_data.keys()), list(train_data.values()))
