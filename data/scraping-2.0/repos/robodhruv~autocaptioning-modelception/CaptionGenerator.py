"""
Caption generation process 
1. (region proposal) Run GLIP to detect relevant objects and get crops using the bounding boxes
2. (caption per region) Use BLIP2 to generate captions for each crop
3. (filtering) Get the CLIP similarities between every caption and every crop and choose the best caption for each crop
4. (filtering) Throw away the captions that do not have the highest cosine similarities with their crop
5. (captioning) Ask GPT4 to remove any outliers
6. (captioning) Ask GPT4 to generate a final caption for the entire image

Save all the meta generated along the way for possible alternitive downsteam uses and analysis
"""

from PIL import Image, ImageDraw
import os
import time
import torch
import numpy as np
import dotenv
from collections import Counter
from dataclasses import dataclass
from retry import retry
import random

from lavis.models import load_model_and_preprocess
import clip
import openai
from glip_client import GLIPClient

# This is the GPTx prompt, feel free to play around with it
SYSTEM_PROMPT = "Your goal is to take in a series of noisy observations from an image and synthesize a clear picture of the room in one sentence. Ignore unlikeley captions that are outliers."

@dataclass
class ImageCaption:
    caption: str
    crops: list
    labels: list
    captions: list
    remapped_captions: list
    objects_counts: Counter

    def __str__(self) -> str:
        return self.caption

    def __repr__(self) -> str:
        return f"ImageCaption(caption={self.caption}, crops={self.crops}, labels={self.labels}, captions={self.captions}, remapped_captions={self.remapped_captions}, objects_counts={self.objects_counts})"
    
class CaptionGenerator:
    def __init__(self, openai_key, device, verbose=False, filter=True, topk=-1, gpt4=False) -> None:
        self.device = device
        self.words = ["door", "chair", "window", "cabinet", "table", "picture", "cushion", "sofa", "bed", "chest of drawers", "plant", "sink", "toilet", "stool", "towel", "tv monitor", "shower", "bathtub", "counter", "fireplace", "gym equipment", "seating", "clothes", "washing machine", "dishwasher", "staircase", "rug", "range hood"]
        self.verbose = verbose
        self.filter = filter
        self.topk = topk
        self.gpt4 = gpt4
        if self.verbose:
            print("CaptionGenerator VERBOSE")
            print("Filtering TRUE")
            print("Topk", self.topk)
            print("Device", self.device)
            if self.gpt4:
                print("Using GPT4")
            else:
                print("Using GPT3.5")

        openai.api_key = openai_key

        if self.gpt4:
            self.gpt_model = "gpt-4"
        else: 
            self.gpt_model = "gpt-3.5-turbo"
        self.blip_model, self.vis_processors, _ = load_model_and_preprocess(name="blip_caption", model_type="base_coco", is_eval=True, device=self.device)
        self.clip_model, self.clip_preprocess = clip.load("ViT-L/14", device=self.device)
        self.glip_client = GLIPClient(server_url="http://thestral.bair.berkeley.edu:1027/process_image", threshold=0.5)

    def set_words(self, words):
        self.words = words

    def get_embedding(self, text, model="text-embedding-ada-002"):
        text = text.replace("\n", " ")
        return openai.Embedding.create(input = [text], model=model)['data'][0]['embedding']
    
    @retry(tries=3, delay=1, backoff=2)
    def call_gpt(self, system_message, text):
        if self.verbose: print("Calling GPT")
        response = openai.ChatCompletion.create(
        model=self.gpt_model,
        messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": text}
            ]
        )
        return response["choices"][0]["message"]["content"]

    
    def generate_caption(self, image):
        if self.verbose: print("Generating caption for crop")
        image = self.vis_processors["eval"](image).unsqueeze(0).to(self.device)
        return self.blip_model.generate({"image": image})[0]
    
    def get_objects(self, image):
        if self.verbose: print("Getting objects")
        return self.glip_client.get_object_counts(image, self.words)
    
    def get_clip_similarity(self, text, image):
        if self.verbose: print("Getting clip similarity")
        with torch.no_grad():
            text_features = self.clip_model.encode_text(clip.tokenize([text]).to(self.device)).detach().cpu().numpy()
            image_features = self.clip_model.encode_image(self.clip_preprocess(image).unsqueeze(0).to(self.device)).detach().cpu().numpy()
            # Cosine similarity
            return np.dot(text_features, image_features.T).squeeze() / np.linalg.norm(text_features) / np.linalg.norm(image_features)
        
    def get_crops(self, image, bboxes, buffer=50):
        if self.verbose: print("Getting crops")
        object_crops = []
        for bbox in bboxes:
            # use bugger of 150 in this case
            # crop = image.crop((bbox[0] - buffer, bbox[1] - buffer, bbox[0] + buffer, bbox[1] + buffer))
            ## bbox is (x1 y1 x2 y2)
            crop = image.crop((bbox[0] - buffer, bbox[1] - buffer, bbox[2] + buffer, bbox[3] + buffer))
            object_crops.append(crop)
        return object_crops
    
    def caption_image(self, image, verbose=False) -> ImageCaption:
        counts, bboxes, text_labels = self.get_objects(image)

        if self.topk > 0:
            bbox_areas = []
            for bbox in bboxes:
                x1, y1, x2, y2 = bbox
                area = (x2 - x1) * (y2 - y1)
                bbox_areas.append(area)

            # Sort bboxes and areas in descending order by area
            sorted_bboxes_areas = sorted(zip(bboxes, bbox_areas), key=lambda x: x[1], reverse=True)

            # Get the top k bboxes
            k = self.topk  # Change k to any number you want
            top_k_bboxes = [bbox for bbox, area in sorted_bboxes_areas[:k]]
            bboxes = top_k_bboxes

        crops = self.get_crops(image, bboxes)
        captions = []
        for crop in crops:
            captions.append(self.generate_caption(crop))
        
        if self.filter:
            remapped_captions = []
            for crop in crops:
                similarities = []
                for caption in captions:
                    similarities.append(self.get_clip_similarity(caption, crop))
                max_index = np.argmax(similarities)
                remapped_captions.append(captions[max_index])
        else:    
            remapped_captions = captions

        # Use GPT to give a single final captions
        final_caption = ""
        for caption in remapped_captions:
            final_caption += caption + " "

        final_caption = self.call_gpt(SYSTEM_PROMPT, final_caption)

        # Construct the image caption object
        image_caption = ImageCaption(
            caption=final_caption,
            crops=crops,
            labels=text_labels,
            captions=captions,
            remapped_captions=remapped_captions,
            objects_counts=counts
        )

        return image_caption
    

if __name__ == "__main__":
    # Load a test image 
    image = Image.open("test_images/house_kitchen.png")
    if image.mode == 'RGBA':
        image = image.convert('RGB')


    # Load the caption generator
    print("Loading caption generator")
    dotenv.load_dotenv(".env", override=True)
    openai_key =  os.getenv("OPENAI_API_KEY")
    device = 'cuda:0'
    cg = CaptionGenerator(openai_key, device, verbose=True, filter=False, topk=5)


    # Generate a caption
    caption = cg.caption_image(image)
    print(caption)

    if not os.path.exists("test_images"):
        os.makedirs("test_images")
    draw = ImageDraw.Draw(image)
    draw.text((0, 0), caption.caption, (0, 0, 0), size=20)
    if not os.path.exists("test_images"):
        os.makedirs("test_images")
    image.save(f"test_images/cg_output.png")
    
