import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import (
    Input, Embedding, LSTM, Dense, Dropout, 
    GlobalAveragePooling2D, Add, Concatenate, 
    BatchNormalization, RepeatVector  # Explicitly include RepeatVector
)
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Sequential


def build_custom_cnn(input_shape=(224, 224, 3)):
    """
    Build a custom CNN for feature extraction (Lightweight).
    Args:
        input_shape (tuple): Shape of the input images.
    Returns:
        tf.keras.Model: Custom CNN model.
    """
    model = Sequential([
        tf.keras.layers.Conv2D(16, (3, 3), activation='relu', padding='same', input_shape=input_shape),
        tf.keras.layers.MaxPooling2D((2, 2)),

        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        tf.keras.layers.MaxPooling2D((2, 2)),

        tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        GlobalAveragePooling2D(),

        Dense(128, activation='relu'),
        BatchNormalization(),
        Dropout(0.3)
    ])
    return model


def build_pretrained_cnn(input_shape=(224, 224, 3)):
    """
    Build a feature extractor using a pre-trained ResNet50.
    Args:
        input_shape (tuple): Shape of the input images.
    Returns:
        tf.keras.Model: Pre-trained CNN model.
    """
    base_model = ResNet50(include_top=False, weights='imagenet', input_shape=input_shape)
    base_model.trainable = False  # Freeze the base model
    model = Sequential([
        base_model,
        GlobalAveragePooling2D(),
        Dense(128, activation='relu'),
        BatchNormalization(),
        Dropout(0.3)
    ])
    return model


def build_captioning_model(vocab_size, max_sequence_length, feature_extractor='pretrained', input_shape=(224, 224, 3)):
    """
    Build the image captioning model.
    """
    # Feature extraction
    if feature_extractor == 'custom':
        image_model = build_custom_cnn(input_shape)
    else:
        image_model = build_pretrained_cnn(input_shape)

    # Image feature input
    image_input = Input(shape=input_shape, name="image_input")
    image_features = image_model(image_input)  # Shape: (batch_size, 128)

    # Text input
    text_input = Input(shape=(None,), name="text_input")
    text_embeddings = Embedding(input_dim=vocab_size, output_dim=128, mask_zero=True)(text_input)
    text_features = LSTM(128, return_sequences=True)(text_embeddings)  # Shape: (batch_size, seq_length, 128)

    # Expand image features to match text sequence length
    # image_features_expanded = RepeatVector(max_sequence_length - 1)(image_features)  # Shape: (batch_size, seq_length-1, 128)
    image_features_expanded = RepeatVector(max_sequence_length)(image_features)  # Ensure both tensors have the same sequence length


    # Fusion
    combined_features = Concatenate(axis=-1)([image_features_expanded, text_features])  # Shape: (batch_size, seq_length-1, 256)

    # Output layers
    dense = Dense(128, activation='relu')(combined_features)
    dropout = Dropout(0.3)(dense)
    outputs = Dense(vocab_size, activation='softmax')(dropout)

    # Build and return the model
    model = Model(inputs=[image_input, text_input], outputs=outputs)
    return model



if __name__ == "__main__":
    # Example usage
    VOCAB_SIZE = 7758  # Example vocabulary size
    MAX_SEQUENCE_LENGTH = 20  # Example sequence length

    # # Build model with a custom CNN
    # model_custom = build_captioning_model(VOCAB_SIZE, MAX_SEQUENCE_LENGTH, feature_extractor='custom')
    # model_custom.summary()

    # Build model with a pretrained CNN
    model_pretrained = build_captioning_model(VOCAB_SIZE, MAX_SEQUENCE_LENGTH, feature_extractor='pretrained')
    model_pretrained.summary()
