# import os
# import numpy as np
# import tensorflow as tf
# import pickle
# import logging
# from src.model import build_captioning_model

# # Setup logging
# logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")

# # Paths
# PREPROCESSED_DIR = "data/preprocessed/"
# MODEL_SAVE_DIR = "models/"
# os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

# # Optimized Training Parameters
# BATCH_SIZE = 4  # Lowered to reduce memory usage
# EPOCHS = 5      # Start with fewer epochs to test stability
# LIMIT = 1500    # Use a smaller dataset for debugging

# def load_tokenizer():
#     """ Load tokenizer from preprocessed data """
#     with open(os.path.join(PREPROCESSED_DIR, "tokenizer.pkl"), "rb") as f:
#         tokenizer = pickle.load(f)
#     return tokenizer

# def load_limited_data(limit=1500):
#     """
#     Load a subset of the preprocessed dataset for debugging.
#     """
#     logging.info(f"Loading a limited dataset: {limit} samples")

#     # Training data
#     train_images = np.load(os.path.join(PREPROCESSED_DIR, "train_images.npy"), mmap_mode='r')[:limit]
#     train_captions = np.load(os.path.join(PREPROCESSED_DIR, "train_captions.npy"), mmap_mode='r')[:limit * 5]

#     # Validation data
#     val_images = np.load(os.path.join(PREPROCESSED_DIR, "val_images.npy"), mmap_mode='r')[:limit // 5]
#     val_captions = np.load(os.path.join(PREPROCESSED_DIR, "val_captions.npy"), mmap_mode='r')[: (limit // 5) * 5]  # Ensure 5 captions per image

#     # Expand images to match captions (5 per image)
#     train_images = np.repeat(train_images, 5, axis=0)
#     val_images = np.repeat(val_images, 5, axis=0)

#     logging.info(f"Dataset Shapes -> Train Images: {train_images.shape}, Train Captions: {train_captions.shape}")
#     logging.info(f"Validation Shapes -> Val Images: {val_images.shape}, Val Captions: {val_captions.shape}")

#     return train_images, train_captions, val_images, val_captions

# def prepare_labels(captions):
#     """
#     Shift captions to create input and output labels for training.
#     Args:
#         captions (np.ndarray): Tokenized captions.
#     Returns:
#         tuple: input_captions and output_labels
#     """
#     # input_captions = captions[:, :-1]  # Shape: (batch_size, max_seq_length-1)
#     # output_labels = captions[:, 1:]    # Shape: (batch_size, max_seq_length-1)

#     input_captions = captions[:, :]  # Keep full sequence length
#     output_labels = captions[:, 1:]  # Shift for next-word prediction


#     logging.info(f"Prepared Labels -> Input Captions: {input_captions.shape}, Output Labels: {output_labels.shape}")

#     return input_captions, output_labels

# def create_tf_dataset(images, input_captions, output_labels, batch_size):
#     """
#     Create a TensorFlow dataset for efficient batch processing.
#     Args:
#         images (np.ndarray): Array of images.
#         input_captions (np.ndarray): Input tokenized captions.
#         output_labels (np.ndarray): Output labels (shifted captions).
#         batch_size (int): Batch size.
#     Returns:
#         tf.data.Dataset: TensorFlow dataset.
#     """
#     dataset = tf.data.Dataset.from_tensor_slices(
#         ({"image_input": images, "text_input": input_captions}, output_labels)
#     )
#     dataset = dataset.shuffle(len(images))
#     dataset = dataset.batch(batch_size, drop_remainder=True).prefetch(buffer_size=tf.data.AUTOTUNE)
#     return dataset

# def main():
#     logging.info("Starting training process...")

#     # Load tokenizer
#     tokenizer = load_tokenizer()
#     vocab_size = len(tokenizer.word_index) + 1
#     max_sequence_length = 20

#     # Load a limited dataset for debugging
#     train_images, train_captions, val_images, val_captions = load_limited_data(limit=LIMIT)

#     # Prepare labels
#     train_input_captions, train_output_labels = prepare_labels(train_captions)
#     val_input_captions, val_output_labels = prepare_labels(val_captions)

#     logging.info(f"Final Shapes -> Train Captions: {train_input_captions.shape}, Train Labels: {train_output_labels.shape}")
#     logging.info(f"Validation Shapes -> Val Captions: {val_input_captions.shape}, Val Labels: {val_output_labels.shape}")

#     # Create TensorFlow datasets
#     logging.info("Creating TensorFlow datasets...")
#     train_dataset = create_tf_dataset(train_images, train_input_captions, train_output_labels, BATCH_SIZE)
#     val_dataset = create_tf_dataset(val_images, val_input_captions, val_output_labels, BATCH_SIZE)

#     # Train both "custom" and "pretrained" models sequentially
#     for feature_extractor in ["custom", "pretrained"]:
#         logging.info(f"Building and training model with {feature_extractor} CNN...")

#         # Build and compile model
#         model = build_captioning_model(vocab_size, max_sequence_length, feature_extractor=feature_extractor)
#         model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
#         model.summary()

#         # Train the model
#         logging.info(f"Starting model training for {feature_extractor} CNN...")
#         history = model.fit(
#             train_dataset,
#             validation_data=val_dataset,
#             epochs=EPOCHS
#         )

#         # Save model and history
#         model.save(os.path.join(MODEL_SAVE_DIR, f"image_captioning_model_{feature_extractor}.h5"))
#         np.save(os.path.join(MODEL_SAVE_DIR, f"training_history_{feature_extractor}.npy"), history.history)
#         logging.info(f"Training completed and model saved for {feature_extractor} CNN.")

# if __name__ == "__main__":
#     main()



import os
import numpy as np
import tensorflow as tf
import pickle
import logging
from src.model import build_captioning_model

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")

# Paths
PREPROCESSED_DIR = "data/preprocessed/"
MODEL_SAVE_DIR = "models/"
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

# Training Parameters
BATCH_SIZE = 4
EPOCHS = 10
LIMIT = 1000  # Increased dataset size

def load_tokenizer():
    with open(os.path.join(PREPROCESSED_DIR, "tokenizer.pkl"), "rb") as f:
        return pickle.load(f)

def load_limited_data(limit=5000):
    logging.info(f"Loading dataset: {limit} samples")

    train_images = np.load(os.path.join(PREPROCESSED_DIR, "train_images.npy"), mmap_mode='r')[:limit]
    train_captions = np.load(os.path.join(PREPROCESSED_DIR, "train_captions.npy"), mmap_mode='r')[:limit * 5]

    val_images = np.load(os.path.join(PREPROCESSED_DIR, "val_images.npy"), mmap_mode='r')[:limit // 5]
    val_captions = np.load(os.path.join(PREPROCESSED_DIR, "val_captions.npy"), mmap_mode='r')[:(limit // 5) * 5]

    train_images = np.repeat(train_images, 5, axis=0)
    val_images = np.repeat(val_images, 5, axis=0)

    return train_images, train_captions, val_images, val_captions

def train_model(feature_extractor):
    logging.info(f"Training model with {feature_extractor} CNN...")
    
    tokenizer = load_tokenizer()
    vocab_size = len(tokenizer.word_index) + 1
    max_sequence_length = 20

    train_images, train_captions, val_images, val_captions = load_limited_data(LIMIT)

    model = build_captioning_model(vocab_size, max_sequence_length, feature_extractor=feature_extractor)
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    
    model.fit(
        {"image_input": train_images, "text_input": train_captions},
        train_captions,
        validation_data=({"image_input": val_images, "text_input": val_captions}, val_captions),
        batch_size=BATCH_SIZE,
        epochs=EPOCHS
    )

    model.save(os.path.join(MODEL_SAVE_DIR, f"image_captioning_model_{feature_extractor}.h5"))
    logging.info(f"Model saved: image_captioning_model_{feature_extractor}.h5")

if __name__ == "__main__":
    # train_model("custom")

    logging.info("Skipping training for 'custom' (already trained).")
    train_model("pretrained")  # Only train the ResNet model

