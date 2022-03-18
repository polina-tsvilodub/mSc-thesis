import tensorflow as tf 
import os 

def load_image(
        path: str,
        size=None,
    ) -> tf.image:
    """
    Load the image from the given file-path and resize it
    to the given size if not None.

    Arguments
    --------
        path: str
            Path to single image
        size: int
            Size of image to be used (optional)

    Returns
    ------
        img: tf.image 
            Single preprocessed image with tf.float32 encoding           
    """
    # Load image from given location
    path_image = os.path.join("../../../data/val/zips/val2014.zip/", path)
    img = tf.io.read_file(path_image)
    img = tf.image.decode_jpeg(img, channels=3)
    # Resize image if desired
    if not size is None:
        img = tf.image.resize(img, IMAGE_SIZE)
    
    img = tf.image.convert_image_dtype(img, tf.float32)
    # Scale image-pixels so they fall between 0.0 and 1.0
    img = img / 255.0

    return img