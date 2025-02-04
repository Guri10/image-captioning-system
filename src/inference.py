# import os
# import numpy as np
# import tensorflow as tf
# import pickle
# import logging
# from PIL import Image
# import matplotlib.pyplot as plt
# from tensorflow.keras.preprocessing.sequence import pad_sequences
# from src.model import build_captioning_model

# from tensorflow.keras.models import load_model

# # Fix the model by reloading and saving it again
# def reload_and_fix_model(model_path):
#     """
#     Reloads the trained model with the fixed structure and re-saves it.
#     This ensures that the updated `model.py` structure is used.
#     Args:
#         model_path (str): Path to the saved model file.
#     """
#     print(f"Reloading trained model from: {model_path}")
#     model = load_model(model_path, compile=False)  # Load trained weights into the fixed model
#     model.save(model_path)  # Save updated structure
#     print(f"Model structure fixed and saved back to: {model_path}")

# # Run this only once after fixing `model.py`
# reload_and_fix_model("models/image_captioning_model_custom.h5")
# reload_and_fix_model("models/image_captioning_model_pretrained.h5")  # If using ResNet


# # Setup logging
# logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")

# # Paths
# MODEL_SAVE_DIR = "models/"
# PREPROCESSED_DIR = "data/preprocessed/"
# IMAGE_INPUT_DIR = "/Users/atharvagurav/Documents/image_captioning_project/image-captioning-system/data/test"  # Directory where test images are stored

# # Load tokenizer
# with open(os.path.join(PREPROCESSED_DIR, "tokenizer.pkl"), "rb") as f:
#     tokenizer = pickle.load(f)

# print("Tokenizer Word Index Sample:", list(tokenizer.word_index.keys())[:20])  # Print first 20 words


# vocab_size = len(tokenizer.word_index) + 1
# max_sequence_length = 20  # Must match training setting

# def preprocess_image(image_path):
#     """
#     Load and preprocess an image for model inference.
#     Args:
#         image_path (str): Path to the input image.
#     Returns:
#         np.ndarray: Preprocessed image array.
#     """
#     image = Image.open(image_path).convert("RGB")
#     image = image.resize((224, 224))  # Ensure it matches training size
#     image = np.array(image) / 255.0  # Normalize
#     image = np.expand_dims(image, axis=0)  # Add batch dimension
#     return image

# def generate_caption(model, image, tokenizer):
#     start_token = tokenizer.word_index["<start>"]
#     end_token = tokenizer.word_index["<end>"]
#     vocab_size = len(tokenizer.word_index) + 1  # Ensure vocab size matches

#     caption_sequence = [start_token]

#     for _ in range(max_sequence_length - 1):
#         sequence_padded = pad_sequences([caption_sequence], maxlen=max_sequence_length, padding="post")

#         # predictions = model.predict({"image_input": image, "text_input": sequence_padded}, verbose=0)
#         predictions = model.predict([image, sequence_padded], verbose=0)  # Pass inputs as a list

#         predicted_id = np.argmax(predictions[0, len(caption_sequence) - 1, :])  # Get predicted word index


#         # Ensure the predicted index is within valid range
#         # predicted_id = min(predicted_id, vocab_size - 1)
#         predicted_id = max(1, min(predicted_id, vocab_size - 1))  # Ensure index is within vocab range


#         if predicted_id == end_token:
#             break

#         caption_sequence.append(predicted_id)

#     reverse_word_map = {idx: word for word, idx in tokenizer.word_index.items()}
#     generated_caption = " ".join([reverse_word_map.get(idx, "<unk>") for idx in caption_sequence[1:]])

#     return generated_caption


# def run_inference(model_type="custom"):
#     """
#     Run inference on a sample image using a trained model.
#     Args:
#         model_type (str): "custom" for Custom CNN, "pretrained" for ResNet50.
#     """
#     # Load trained model
#     model_path = os.path.join(MODEL_SAVE_DIR, f"image_captioning_model_{model_type}.h5")
#     logging.info(f"Loading model: {model_path}")
#     # model = tf.keras.models.load_model(model_path, custom_objects={"build_captioning_model": build_captioning_model})
#     # Load trained model and compile it
#     model = tf.keras.models.load_model(model_path, custom_objects={"build_captioning_model": build_captioning_model})
#     model.compile(optimizer="adam", loss="sparse_categorical_crossentropy")


#     # Select an image for inference
#     test_images = [f for f in os.listdir(IMAGE_INPUT_DIR) if f.endswith((".jpg", ".png"))]
#     if not test_images:
#         logging.error("No test images found in directory!")
#         return

#     image_path = os.path.join(IMAGE_INPUT_DIR, test_images[0])  # Pick the first image
#     logging.info(f"Running inference on: {image_path}")

#     # Preprocess the image
#     image = preprocess_image(image_path)

#     # Ensure <start> and <end> tokens exist in tokenizer
#     if "<start>" not in tokenizer.word_index:
#         tokenizer.word_index["<start>"] = max(tokenizer.word_index.values()) + 1
#     if "<end>" not in tokenizer.word_index:
#         tokenizer.word_index["<end>"] = max(tokenizer.word_index.values()) + 1

#     print("Updated Tokenizer Word Index Sample:", list(tokenizer.word_index.keys())[:25])


#     # Generate a caption
#     caption = generate_caption(model, image, tokenizer)

#     # Display the image and caption
#     plt.figure(figsize=(8, 6))
#     plt.imshow(Image.open(image_path))
#     plt.title(f"Generated Caption: {caption}")
#     plt.axis("off")
#     plt.show()

# if __name__ == "__main__":
#     run_inference(model_type="custom")  # Change to "pretrained" for ResNet model



import os
import numpy as np
import pickle
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from model import build_captioning_model  # Import the model architecture

# Paths
DATA_DIR = "data/flickr8k/"
OUTPUT_DIR = "data/preprocessed/"
MODEL_PATH = "/Users/atharvagurav/Documents/image_captioning_project/image-captioning-system/models/image_captioning_model_pretrained.h5"  # Path to trained model weights
TOKENIZER_PATH = os.path.join(OUTPUT_DIR, "tokenizer.pkl")

# Model Parameters
VOCAB_SIZE = 8495  # Must match training
MAX_SEQUENCE_LENGTH = 20  # Must match training
IMAGE_SHAPE = (224, 224, 3)
FEATURE_EXTRACTOR = 'pretrained'  # Change to 'custom' if using custom CNN

# Load Tokenizer
with open(TOKENIZER_PATH, "rb") as f:
    tokenizer = pickle.load(f)

# Load Trained Model
model = build_captioning_model(VOCAB_SIZE, MAX_SEQUENCE_LENGTH, FEATURE_EXTRACTOR, IMAGE_SHAPE)
model.layers[1].build((None, VOCAB_SIZE))
model.load_weights(MODEL_PATH)

def preprocess_image(image_path):
    """Load and preprocess an image."""
    image = load_img(image_path, target_size=(224, 224))
    image = img_to_array(image) / 255.0  # Normalize
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

def generate_caption(image_path):
    """Generate a caption using greedy search decoding."""
    image = preprocess_image(image_path)
    caption = ['<start>']
    
    for _ in range(MAX_SEQUENCE_LENGTH - 1):
        sequence = tokenizer.texts_to_sequences([caption])[0]
        sequence = pad_sequences([sequence], maxlen=MAX_SEQUENCE_LENGTH, padding='post')
        
        predictions = model.predict([image, sequence], verbose=0)
        predicted_id = np.argmax(predictions[0, -1, :])
        predicted_word = tokenizer.index_word.get(predicted_id, '<unk>')
        
        if predicted_word == '<end>':
            break
        caption.append(predicted_word)
    
    return ' '.join(caption[1:])  # Remove <start>

if __name__ == "__main__":
    test_image_path = os.path.join(DATA_DIR, "Images", "//Users/atharvagurav/Documents/image_captioning_project/image-captioning-system/data/test/44129946_9eeb385d77.jpg")  # Change to test image
    caption = generate_caption(test_image_path)
    print("Generated Caption:", caption)