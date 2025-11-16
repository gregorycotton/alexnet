import torch
import torch.nn.functional as F
from PIL import Image
import pickle
import sys

from model import AlexNet

MODEL_PATH = "alexnet_cifar10.pth"
DATA_DIR = "./data/cifar-10-batches-py"

# Load class names
def load_class_names(meta_path):
    class_names = []
    try:
        with open(meta_path, 'rb') as f:
            meta = pickle.load(f, encoding='bytes')
            class_names = [name.decode('utf-8') for name in meta[b'label_names']]
    except Exception as e:
        print(f"Error loading class names from {meta_path}: {e}")
    return class_names

# Prepare image
def preprocess_image(image_path):
    normalize_mean = torch.tensor([0.485, 0.456, 0.406])
    normalize_std = torch.tensor([0.229, 0.224, 0.225])

    try:
        image = Image.open(image_path).convert("RGB")
    except Exception as e:
        print(f"Error opening image {image_path}: {e}")
        return None

    # Resize to 224x224
    image = image.resize((224, 224))
    
    # Convert to tensor, normalize to [0, 1]
    r, g, b = image.split()
    r_data = list(r.getdata())
    g_data = list(g.getdata())
    b_data = list(b.getdata())
    
    img_tensor = torch.tensor([r_data, g_data, b_data], dtype=torch.float32).view(3, 224, 224)
    img_tensor /= 255.0

    mean = normalize_mean.view(3, 1, 1)
    std = normalize_std.view(3, 1, 1)
    img_tensor = (img_tensor - mean) / std
    
    return img_tensor.unsqueeze(0)


def predict(image_path):
    print(f"Loading model from {MODEL_PATH}...")
    
    # Load class names
    class_names = load_class_names(f"{DATA_DIR}/batches.meta")
    if not class_names:
        print("Could not load class names. Exiting.")
        return

    # Brand new untrained model yipee
    model = AlexNet(num_classes=10)
    
    # Load my weights
    try:
        model.load_state_dict(torch.load(MODEL_PATH))
    except Exception as e:
        print(f"Error loading model weights: {e}")
        return

    model.eval()

    # Image preprocess
    print(f"Processing image: {image_path}...")
    image_tensor = preprocess_image(image_path)
    if image_tensor is None:
        return
        
    # Prediction
    with torch.no_grad():
        outputs = model(image_tensor)
        
        # Convert logits to probabilities
        probabilities = F.softmax(outputs, dim=1)
        
        # Top 3 predictions
        top3_prob, top3_indices = torch.topk(probabilities, 3)

    print("\nTop 3 Predictions")
    for i in range(top3_indices.size(1)):
        prob = top3_prob[0][i].item() * 100
        class_index = top3_indices[0][i].item()
        class_name = class_names[class_index]
        print(f"  {i+1}. {class_name}: {prob:.2f}%")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python3 predict.py <path_to_image>")
    else:
        predict(sys.argv[1])