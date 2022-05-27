import clip
import os
import numpy as np
import torch
from transformers import GPT2Tokenizer
import skimage.io as io
from PIL import Image

from models.CLIPCap import ClipCaptionModel, generate_beam, generate2

# @param ['COCO', 'Conceptual captions']
# pretrained_model = 'COCO'  
pretrained_model = 'Conceptual captions'

model_path = '../weights/conceptual_weights.pt' if pretrained_model == 'Conceptual captions' else '../weights/coco_weights.pt'

is_gpu = torch.cuda.is_available()
device = 'cuda' if is_gpu else "cpu"
clip_model, preprocess = clip.load("ViT-B/32", device=device, jit=False)
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

prefix_length = 10

model = ClipCaptionModel(prefix_length)

model.load_state_dict(torch.load(model_path, map_location='cpu')) 

model = model.eval() 
device = 'cuda' if is_gpu else "cpu"
model = model.to(device)

def get_image_prefix(img, clip_model):
    """ Get CLIP emb of entire dataset and average them """
    prefix = clip_model.encode_image(img).to(device, dtype=torch.float32)
    return prefix

def caption_image(img_dir):
    pil_image = Image.open(img_dir)
    image = preprocess(pil_image).unsqueeze(0).to(device)
    with torch.no_grad():
        prefix = get_image_prefix(image, clip_model=clip_model)
        prefix_embed = model.clip_project(prefix).reshape(1, prefix_length, -1)

    generated_text_prefix = generate2(model, tokenizer, embed=prefix_embed)

    return generated_text_prefix

# This is the most important... we'll be generating vectors to perform search with.
def generate_text_embedding(text):
    with torch.no_grad():
        tokenized_text = clip.tokenize(text).to(device)
        prefix = clip_model.encode_text(tokenized_text).to(device, dtype=torch.float32)
        embedding = model.clip_project(prefix).reshape(1, prefix_length, -1)
    
    return embedding

def generate_embedding_text(embed):
    generated_text_prefix = generate2(model, tokenizer, embed=embed)
    return generated_text_prefix