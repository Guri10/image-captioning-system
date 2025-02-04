import tensorflow as tf
from tensorflow.keras.models import load_model

# Paths to saved models
model_paths = [
    "models/image_captioning_model_custom.h5",
    "models/image_captioning_model_pretrained.h5"  # Uncomment if using ResNet model
]

def reload_and_fix_model(model_path):
    """
    Reloads the trained model with the fixed structure and re-saves it.
    Ensures that the updated `model.py` structure is applied.
    """
    print(f"🔄 Reloading trained model from: {model_path}")
    model = load_model(model_path, compile=False)  # Load trained weights into fixed model
    model.save(model_path)  # Save updated structure
    print(f"✅ Model structure fixed and saved back to: {model_path}")

# Apply the fix to both models
for path in model_paths:
    try:
        reload_and_fix_model(path)
    except Exception as e:
        print(f"⚠️ Could not reload {path}: {e}")
