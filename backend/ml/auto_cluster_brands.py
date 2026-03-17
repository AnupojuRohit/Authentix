import os
import torch
import open_clip
from PIL import Image
import numpy as np
from tqdm import tqdm
import shutil

# Paths
INPUT_DIR = "../../dataset/raw"
OUTPUT_DIR = "../../dataset/clustered_by_brand"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Define a comprehensive list of fashion, sportswear, and luxury brands
# CLIP will match the images against this list. Add more here if your dataset has specific niche brands.
BRANDS = [
    "Nike", "Adidas", "Gucci", "Louis Vuitton", "Supreme", "Puma", 
    "Balenciaga", "Off-White", "New Balance", "Versace", "Ralph Lauren", 
    "Stone Island", "Prada", "Fendi", "Burberry", "Converse", "Vans",
    "Under Armour", "Reebok", "Givenchy", "Champion", "Timberland",
    "Calvin Klein", "Tommy Hilfiger", "Lacoste", "Asics", "Fila",
    "Jordan", "Yeezy", "Hermes", "Chanel", "Dior", "Rolex", "Cartier",
    "Bottega Veneta", "Celine", "Saint Laurent", "Moncler", "Alexander McQueen",
    "Valentino", "Miu Miu", "Goyard", "Kenzo", "Maison Margiela", "Rick Owens",
    "Fear of God", "Bathing Ape", "Stussy", "Kith", "Palace", "Patagonia", 
    "The North Face", "Columbia", "Arc'teryx", "Salomon", "Hoka"
]

# Create text prompts for zero-shot classification
text_prompts = [f"a photo of a {brand} product, logo, or clothing" for brand in BRANDS]

print("Loading OpenCLIP Model...")
device = "cuda" if torch.cuda.is_available() else "cpu"
model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='openai')
model = model.eval().to(device)
tokenizer = open_clip.get_tokenizer('ViT-B-32')

print("Encoding Brand Text Prompts...")
text_tokens = tokenizer(text_prompts).to(device)
with torch.no_grad():
    text_features = model.encode_text(text_tokens)
    text_features /= text_features.norm(dim=-1, keepdim=True)

print(f"Starting auto-sorting of images from {INPUT_DIR}...")
files = []
for root_dir, dirs, filenames in os.walk(INPUT_DIR):
    for f in filenames:
        if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp')):
            files.append(os.path.join(root_dir, f))

for path in tqdm(files):
    file = os.path.basename(path)

    try:
        # Load and preprocess image
        image = Image.open(path).convert("RGB")
        image_input = preprocess(image).unsqueeze(0).to(device)

        # Extract image features
        with torch.no_grad():
            image_features = model.encode_image(image_input)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            
            # Compute cosine similarity between image and all text prompts
            # Multiply by 100 to scale the softmax nicely
            text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)
            
        # Get the top predicted brand
        top_prob, top_idx = text_probs[0].topk(1)
        prob_val = top_prob.item()
        
        # If probability is high enough, it's a confident match, otherwise put in 'Unknown'
        if prob_val > 0.05: # Threshold can be adjusted
            predicted_brand = BRANDS[top_idx.item()]
        else:
            predicted_brand = "Unknown"

    except Exception as e:
        predicted_brand = "Error_Processing"

    # Create folder for the detected brand
    brand_folder_name = predicted_brand.replace(" ", "_")
    brand_dir = os.path.join(OUTPUT_DIR, brand_folder_name)
    os.makedirs(brand_dir, exist_ok=True)
    
    # Copy file to the new brand folder
    dst = os.path.join(brand_dir, file)
    shutil.copy(path, dst)

print(f"\nSorting complete! Check the '{OUTPUT_DIR}' folder.")