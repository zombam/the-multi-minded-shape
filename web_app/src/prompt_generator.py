import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from collections import defaultdict
from sklearn.cluster import KMeans
import numpy as np

device = "cpu"

class ConvNeuralNetwork(nn.Module):
    def __init__(self):
        # Call the constructor of the base class nn.module
        super().__init__()

        modules = []
        
        # each convolutional layer will takes a batch of 3D tensor 
        # Outputs a batch of 3D tensor (outputs of all convolutional filters)
        # for the first conv layer, the input channel is 3 because that's the number of colour channels
        modules.append(nn.Conv2d(3, 8, kernel_size=3, padding=1))
        
        # Each conv layer will be followed by a pooling layer and the relu activation function
        # the pooling layer reduce widths and heights of the inputs by half 
        # so the tensor shape change from [8, 64, 64] -> [8, 32, 32] after it
        modules.append(nn.MaxPool2d(2, 2))
        modules.append(nn.ReLU())
        
        # the second conv layer
        # we increase the number of channels, but the widths and heights remain the same
        # so the tensor shape change from [8, 32, 32] -> [16, 32, 32] after it
        modules.append(nn.Conv2d(8, 16, kernel_size=3, padding=1))
        
        # the second pooling and relu layer
        # the tensor shape change from [16, 32, 32] -> [16, 16, 16] after it
        modules.append(nn.MaxPool2d(2, 2))
        modules.append(nn.ReLU())
        
        # the tensor shape change from [16, 16, 16] -> [32, 8, 8] after it
        modules.append(nn.Conv2d(16, 32, kernel_size=3, padding=1))
        modules.append(nn.MaxPool2d(2, 2))
        modules.append(nn.ReLU()) 

        # the tensor shape change from [32, 8, 8] -> [32, 4, 4] after it
        modules.append(nn.Conv2d(32, 32, kernel_size=3, padding=1))
        modules.append(nn.MaxPool2d(2, 2))
        modules.append(nn.ReLU()) 
        
        # We append these modules to a Modules List object so that later we can iterate through it
        self.convolutions = nn.ModuleList(modules)
        
        # First fully connected layer
        # Its input size depend on the convolutional layer's output resolution and the number of channels
        self.fc1 = nn.Linear(32 * 14 * 14, 32)
        
        # Last fully connected layer
        # Outputs a vector of class predictions,
        # make sure the output channel is the number of classes you have
        self.fc2 = nn.Linear(32, 47)
        

    # Definition of the forward pass
    # Here the classifier takes an image as input and predicts an vector of probabilites
    def forward(self, x):

        # Pass input through all layers we have added
        for layer in self.convolutions:
            x = layer(x)

        # Flatten the output of the last convolutional layer into a 1-dimensional vector
        x = torch.flatten(x, 1) 
        
        # Pass through the first and the second fully connected layer with relu activation function
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        
        # Output a vector of class probabilities
        return x

# Instantiate and load weights
model = ConvNeuralNetwork()
model.load_state_dict(torch.load("../checkpoints/model_final.pt", map_location=device))
model.eval()

# ---------- SETTINGS ----------
image_path = 'test.jpg'                     # input image
model_path = '../checkpoints/model_final.pt'

# ---------- LOAD MODEL ----------
model = ConvNeuralNetwork()
model.load_state_dict(torch.load(model_path, map_location='cpu'))
model.eval()

# ---------- IMAGE PREPROCESS ----------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

image = Image.open(image_path).convert('RGB')
input_tensor = transform(image).unsqueeze(0)  # Add batch dimension

# ---------- INFERENCE ----------
with torch.no_grad():
    logits = model(input_tensor)
    probs = F.softmax(logits, dim=1).squeeze()

# ---------- CLASS NAMES ----------
idx_to_texture = [
    'banded', 'blotchy', 'braided', 'bubbly', 'bumpy', 'chequered', 'cobwebbed', 'cracked', 'crosshatched',
    'crystalline', 'dotted', 'fibrous', 'flecked', 'freckled', 'frilly', 'gauzy', 'grid', 'grooved',
    'honeycombed', 'interlaced', 'knitted', 'lacelike', 'lined', 'marbled', 'matted', 'meshed', 'paisley',
    'perforated', 'pitted', 'pleated', 'polka-dotted', 'porous', 'potholed', 'scaly', 'smeared', 'spiralled',
    'sprinkled', 'stained', 'stratified', 'striped', 'studded', 'swirly', 'veined', 'waffled', 'woven',
    'wrinkled', 'zigzagged'
]

# ---------- SHAPE CATEGORY MAP ----------
SHAPE_CATEGORY_MAP = {
    'lined': 'lines', 'striped': 'lines', 'crosshatched': 'lines', 'zigzagged': 'lines', 'braided': 'lines',
    'polka-dotted': 'point', 'studded': 'point', 'dotted': 'point', 'sprinkled': 'point',
    'freckled': 'point', 'flecked': 'point',
    'grid': 'grid', 'chequered': 'grid', 'waffled': 'grid', 'honeycombed': 'grid',
    'cobwebbed': 'grid', 'meshed': 'grid',
    'swirly': 'flowing', 'spiralled': 'flowing', 'paisley': 'flowing', 'veined': 'flowing',
    'woven': 'flowing', 'knitted': 'flowing', 'interlaced': 'flowing', 'gauzy': 'flowing', 'lacelike': 'flowing',
    'potholed': 'holes', 'pitted': 'holes', 'perforated': 'holes', 'porous': 'holes',
    'bumpy': 'complex', 'scaly': 'complex', 'crystalline': 'complex', 'wrinkled': 'complex',
    'pleated': 'complex', 'frilly': 'complex', 'stained': 'complex', 'marbled': 'complex', 'blotchy': 'complex'
}

# ---------- TEXTURE GENERATION CATEGORIES ----------
TEXTURE_CATEGORIES = {
    'Natural Surfaces': [
        'veined', 'freckled', 'scaly', 'fibrous', 'marbled', 'stratified', 'crystalline', 'bumpy'
    ],
    'Hazardous Patterns': [
        'zigzagged', 'potholed', 'cracked', 'pitted', 'bubbly', 'smeared', 'blotchy'
    ],
    'Food-Like Textures': [
        'honeycombed', 'waffled', 'dotted', 'swirly', 'braided', 'perforated', 'pleated'
    ],
    'Habitat Features': [
        'grid', 'cobwebbed', 'meshed', 'woven', 'gauzy', 'lacelike', 'matted', 'striped',
        'banded', 'studded', 'polka-dotted', 'wrinkled', 'grooved'
    ]
}

# --- Reverse map texture -> texture category ---
TEXTURE_TO_GROUP = {}
for group, tex_list in TEXTURE_CATEGORIES.items():
    for tex in tex_list:
        TEXTURE_TO_GROUP[tex] = group

# ---------- SHAPE PROBABILITIES ----------
shape_probs = defaultdict(float)
texture_group_probs = defaultdict(float)

for i, texture in enumerate(idx_to_texture):
    prob = probs[i].item()

    shape_cat = SHAPE_CATEGORY_MAP.get(texture)
    if shape_cat:
        shape_probs[shape_cat] += prob

    group = TEXTURE_TO_GROUP.get(texture)
    if group:
        texture_group_probs[group] += prob

# Normalize
shape_total = sum(shape_probs.values())
for k in shape_probs:
    shape_probs[k] /= shape_total

texture_total = sum(texture_group_probs.values())
for k in texture_group_probs:
    texture_group_probs[k] /= texture_total

# Get most likely texture category
top_texture_category = max(texture_group_probs.items(), key=lambda x: x[1])[0]
keywords = TEXTURE_CATEGORIES[top_texture_category]

# ---------- DOMINANT COLOR ----------
def get_dominant_color(image, k=3):
    small_img = image.resize((50, 50))
    np_img = np.array(small_img).reshape(-1, 3)
    kmeans = KMeans(n_clusters=k, n_init='auto')
    kmeans.fit(np_img)
    colors = kmeans.cluster_centers_.astype(int)
    return tuple(colors[0])  # RGB

dominant_rgb = get_dominant_color(image)
color_string = f"rgb({dominant_rgb[0]}, {dominant_rgb[1]}, {dominant_rgb[2]})"

# ---------- COMFYUI PROMPTS ----------
positive_prompt = f"seamless texture, surface pattern, top-down view, {', '.join(keywords)}, {color_string}, high detail, tileable, photorealistic,ocean surface, underwater texture, marine pattern, sea-inspired, coral-like, reef structure, ripples, wet surface, algae texture, bioluminescent detail, sandy seabed, water erosion, kelp pattern, organic flow, blue green palette, barnacle crust, tidepool reflection, shell texture, mollusk skin, aquatic surface, oceanic depth"
negative_prompt = "object, person, animal, background, lighting, shadows, reflections, distortion, text, watermark, blur"

# ---------- OUTPUT ----------
print("✅ SHAPE PROBABILITIES:")
for cat, prob in shape_probs.items():
    print(f"{cat}: {prob:.3f}")

print("\n🎨 TOP TEXTURE GROUP:", top_texture_category)
print("👍 Positive Prompt for ComfyUI:")
print(positive_prompt)
print("\n👎 Negative Prompt:")
print(negative_prompt)