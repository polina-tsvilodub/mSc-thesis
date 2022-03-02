def load_captions_data(
        download_dir: str, 
        filename: str,
    ) -> tuple:
    """
    Loads captions (text) data from json files and maps them to corresponding images.

    Arguments:
    --------
        download_dir: str
            Directory holding annotation files
        filename: str
            Path to the text file containing caption data.

    Returns: tuples
    -------
        ids: tuple
            Image numbers as IDs
        filenames: tuple  
            Filenames of single images
        captions: tuple
            Lists of captions for each image
    """

      # Full path for the data-file.
    path = os.path.join(download_dir, "annotations", filename)

    # Load the json file.
    with open(path, "r", encoding="utf-8") as file:
        data_raw = json.load(file)

    # Convenience variables.
    images = data_raw['images']
    annotations = data_raw['annotations']

    # Initialize the dict for holding our data.
    # The lookup-key is the image-id.
    records = dict()

    # Collect all the filenames for the images.
    for image in images:
        # Get the id and filename for this image.
        image_id = image['id']
        filename = image['file_name'] 

        # Initialize a new data-record.
        record = dict()
        # Initialize an empty list of image-captions
        # which will be filled further below.
        record['captions'] = list()

        # Set the image-filename in the data-record.
        record['filename'] = filename
        # Save the record using the the image-id as the lookup-key.
        records[image_id] = record

    # Collect all the captions for the images.
    for ann in annotations:
        # Get the id and caption for an image.
        image_id = ann['image_id']
        caption = ann['caption']
        
        # preprocess caption
        caption = caption.strip().strip("\n").strip("\t")
        # append start and stop tokens
        caption = "<START> " + caption.strip() + " <END>"
        # TODO : see if this should be saved right away too
        tokens = caption.split()
        

        # Lookup the data-record for this image-id.
        # This data-record should already exist from the loop above.
        record = records[image_id]

        # Append the current caption to the list of captions in the
        # data-record that was initialized in the loop above.
        record['captions'].append(caption)

    # Convert the records-dict to a list of tuples.
    records_list = [(key, record['filename'], record['captions'])
                    for key, record in sorted(records.items())]

    # Convert the list of tuples to separate tuples with the data.
    ids, filenames, captions = zip(*records_list)
    
    # TODO decide what needs to be returned here
    print("--- Number of images: ", len(filenames))
    print("--- Number of caption sets: ", len(captions))

    return ids, filenames, captions
        