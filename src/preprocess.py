# import os
# import numpy as np
# import logging
# from PIL import Image
# from tensorflow.keras.preprocessing.text import Tokenizer
# from tensorflow.keras.preprocessing.sequence import pad_sequences
# import pickle

# # Setup logging
# logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")

# # Paths for the Flickr8k dataset
# IMAGE_DIR = "data/flickr8k/Images/"
# ANNOTATION_FILE = "data/flickr8k/captions.txt"
# OUTPUT_DIR = "data/preprocessed/"

# # Constants
# TARGET_IMAGE_SIZE = (224, 224)  # Resize images to this size
# MAX_VOCAB_SIZE = 5000          # Maximum vocabulary size
# MAX_SEQUENCE_LENGTH = 20       # Maximum caption length
# OOV_TOKEN = "<unk>"
# TRAIN_SPLIT = 0.8              # Fraction of data used for training

# # Ensure the output directory exists
# os.makedirs(OUTPUT_DIR, exist_ok=True)


# def load_flickr_annotations(file_path):
#     """
#     Load annotations from the Flickr8k captions.txt file.
#     Args:
#         file_path (str): Path to the captions.txt file.
#     Returns:
#         dict: Dictionary where keys are image filenames and values are lists of captions.
#     """
#     logging.info(f"Loading annotations from {file_path}")
#     annotations = {}
#     with open(file_path, 'r', encoding="utf-8") as f:
#         next(f)  # Skip header
#         for line in f:
#             parts = line.strip().split(',', 1)  # Split on the first comma
#             image_id, caption = parts[0], parts[1]
#             annotations.setdefault(image_id, []).append(caption)
#     return annotations


# def preprocess_image(image_path):
#     """
#     Preprocess an image by resizing it and normalizing pixel values.
#     Args:
#         image_path (str): Path to the image file.
#     Returns:
#         np.ndarray: Preprocessed image array.
#     """
#     try:
#         logging.info(f"Processing image: {image_path}")
#         image = Image.open(image_path).convert("RGB")
#         image = image.resize(TARGET_IMAGE_SIZE)
#         return np.array(image) / 255.0
#     except Exception as e:
#         logging.error(f"Error processing image {image_path}: {e}")
#         return None


# def split_data(annotations, train_split):
#     """
#     Split annotations into training and validation sets.
#     Args:
#         annotations (dict): Dictionary of image-caption pairs.
#         train_split (float): Fraction of data to use for training.
#     Returns:
#         tuple: Training and validation dictionaries.
#     """
#     image_filenames = list(annotations.keys())
#     split_idx = int(len(image_filenames) * train_split)
#     train_annotations = {k: annotations[k] for k in image_filenames[:split_idx]}
#     val_annotations = {k: annotations[k] for k in image_filenames[split_idx:]}
#     return train_annotations, val_annotations


# def extract_image_features(annotations, image_dir):
#     """
#     Extract image features and pair them with corresponding captions.
#     Args:
#         annotations (dict): Dictionary mapping image IDs to captions.
#         image_dir (str): Directory containing images.
#     Returns:
#         list: Preprocessed images.
#         list: Corresponding captions.
#     """
#     images = []
#     captions = []
#     for image_id, captions_list in annotations.items():
#         image_path = os.path.join(image_dir, image_id)
#         image = preprocess_image(image_path)
#         if image is not None:
#             images.append(image)
#             captions.extend(captions_list)
#     return np.array(images), captions


# def process_captions(captions, tokenizer=None):
#     """
#     Tokenize captions and pad sequences.
#     Args:
#         captions (list): List of captions.
#         tokenizer (Tokenizer): Keras Tokenizer object. If None, a new tokenizer is created.
#     Returns:
#         tokenizer: Tokenizer fitted on the captions.
#         sequences: Padded sequences of tokenized captions.
#     """
#     if tokenizer is None:
#         tokenizer = Tokenizer(num_words=MAX_VOCAB_SIZE, oov_token=OOV_TOKEN)
#         tokenizer.fit_on_texts(captions)
    
#     sequences = tokenizer.texts_to_sequences(captions)
#     padded_sequences = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH, padding='post')
#     return tokenizer, padded_sequences


# def save_data(images, captions, tokenizer, output_dir, split_name):
#     """
#     Save preprocessed images, captions, and tokenizer to disk.
#     Args:
#         images (list): List of preprocessed images.
#         captions (list): List of tokenized captions.
#         tokenizer (Tokenizer): Keras Tokenizer object.
#         output_dir (str): Output directory.
#         split_name (str): Split name (e.g., 'train' or 'val').
#     """
#     os.makedirs(output_dir, exist_ok=True)

#     # Save images
#     np.save(os.path.join(output_dir, f"{split_name}_images.npy"), images)

#     # Save captions
#     np.save(os.path.join(output_dir, f"{split_name}_captions.npy"), captions)

#     # Save tokenizer (only once, during training split)
#     if split_name == "train":
#         with open(os.path.join(output_dir, "tokenizer.pkl"), "wb") as f:
#             pickle.dump(tokenizer, f)
#         logging.info("Tokenizer saved")


# def main():
#     # Load annotations
#     annotations = load_flickr_annotations(ANNOTATION_FILE)

#     # Split data into training and validation sets
#     train_annotations, val_annotations = split_data(annotations, TRAIN_SPLIT)

#     # Preprocess training data
#     logging.info("Processing training data...")
#     train_images, train_captions = extract_image_features(train_annotations, IMAGE_DIR)

#     # Tokenize captions
#     logging.info("Tokenizing captions...")
#     tokenizer, train_sequences = process_captions(train_captions)

#     # Save training data
#     save_data(train_images, train_sequences, tokenizer, OUTPUT_DIR, split_name="train")

#     # Preprocess validation data
#     logging.info("Processing validation data...")
#     val_images, val_captions = extract_image_features(val_annotations, IMAGE_DIR)

#     # Tokenize validation captions using the same tokenizer
#     _, val_sequences = process_captions(val_captions, tokenizer=tokenizer)

#     # Save validation data
#     save_data(val_images, val_sequences, tokenizer, OUTPUT_DIR, split_name="val")


# if __name__ == "__main__":
#     main()




import os
import numpy as np
import pickle
import json
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Paths
DATA_DIR = "data/flickr8k/"
OUTPUT_DIR = "data/preprocessed/"
os.makedirs(OUTPUT_DIR, exist_ok=True)

MAX_VOCAB_SIZE = 8000
MAX_SEQUENCE_LENGTH = 20

def load_flickr_annotations(file_path):
    """
    Load Flickr8k annotations from the captions.txt file.
    """
    annotations = {}
    with open(file_path, 'r', encoding="utf-8") as f:
        next(f)  # Skip header
        for line in f:
            parts = line.strip().split(',', 1)
            image_id, caption = parts[0], parts[1].strip().lower()
            caption = f"<start> {caption} <end>"  # Ensure correct token format
            annotations.setdefault(image_id, []).append(caption)
    return annotations

def process_and_save_data():
    annotations = load_flickr_annotations(os.path.join(DATA_DIR, "captions.txt"))
    
    # Tokenizer
    captions = [caption for captions_list in annotations.values() for caption in captions_list]
    tokenizer = Tokenizer(num_words=MAX_VOCAB_SIZE, oov_token="<unk>")
    tokenizer.fit_on_texts(captions)

    with open(os.path.join(OUTPUT_DIR, "tokenizer.pkl"), "wb") as f:
        pickle.dump(tokenizer, f)

    # Process images and captions
    image_filenames = list(annotations.keys())
    train_size = int(0.8 * len(image_filenames))

    train_annotations = {k: annotations[k] for k in image_filenames[:train_size]}
    val_annotations = {k: annotations[k] for k in image_filenames[train_size:]}

    def save_data(annotations, filename):
        images = []
        captions = []

        for image_id, caption_list in annotations.items():
            image_path = os.path.join(DATA_DIR, "Images", image_id)
            image = tf.keras.preprocessing.image.load_img(image_path, target_size=(224, 224))
            image = tf.keras.preprocessing.image.img_to_array(image) / 255.0
            images.append(image)
            captions.extend(tokenizer.texts_to_sequences(caption_list))

        np.save(os.path.join(OUTPUT_DIR, f"{filename}_images.npy"), np.array(images))
        np.save(os.path.join(OUTPUT_DIR, f"{filename}_captions.npy"), pad_sequences(captions, maxlen=MAX_SEQUENCE_LENGTH, padding="post"))

    save_data(train_annotations, "train")
    save_data(val_annotations, "val")

if __name__ == "__main__":
    process_and_save_data()
