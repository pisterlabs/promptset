import torch
import requests
from PIL import Image
from typing import List
from transformers import Owlv2Processor, Owlv2ForObjectDetection
#import matplotlib.pyplot as plt
import numpy as np
from transformers.utils.constants import OPENAI_CLIP_MEAN, OPENAI_CLIP_STD

class Owl:
    def __init__(self):
        self.device = torch.device("cuda")
        self.model = Owlv2ForObjectDetection.from_pretrained("google/owlv2-base-patch16-ensemble")
        self.model = self.model.to(self.device)
        self.model.eval()
        self.processor = Owlv2Processor.from_pretrained("google/owlv2-base-patch16-ensemble")
    
    def detect(self, image: Image, texts: List, threshold=0.25):
        outputs, inputs = self._detect(image, texts)
        unnormalized_image = self._get_preprocessed_image(inputs.pixel_values)
        target_sizes = torch.Tensor([unnormalized_image.size[::-1]])
        #print(image.size[::-1])
        #target_sizes = torch.Tensor([image.size[::-1]]).to(self.device)
        results = self.processor.post_process_object_detection(outputs=outputs, target_sizes=target_sizes, threshold=threshold)
        boxes, scores, labels = results[0]["boxes"], results[0]["scores"], results[0]["labels"]
        return boxes.cpu().detach().numpy(), scores.cpu().detach().numpy(), labels.cpu().detach().numpy(), unnormalized_image

    '''def detect(self, image: Image, texts: List):
        outputs = self._detect(image, texts)
        boxes, scores, labels = self._post_process(outputs)
        return boxes, scores, labels'''
    
    def _get_preprocessed_image(self, pixel_values):
        pixel_values = pixel_values.squeeze().cpu().numpy()
        unnormalized_image = (pixel_values * np.array(OPENAI_CLIP_STD)[:, None, None]) + np.array(OPENAI_CLIP_MEAN)[:, None, None]
        unnormalized_image = (unnormalized_image * 255).astype(np.uint8)
        unnormalized_image = np.moveaxis(unnormalized_image, 0, -1)
        unnormalized_image = Image.fromarray(unnormalized_image)
        return unnormalized_image

    def _detect(self, image: Image, texts: List):
        #image = image.convert("RGB")
        #show_frame(image)
        #image = image.resize((512, 512))
        print(image.size[::-1])
        inputs = self.processor(text=texts, images=image, return_tensors="pt").to(self.device)
        #inputs = processor(text=texts, images=image, return_tensors="pt")
        with torch.no_grad():
            outputs = self.model(**inputs)
            return outputs, inputs

    def _post_process(self, outputs):
        # Get prediction logits
        logits = torch.max(outputs["logits"][0], dim=-1)
        scores = torch.sigmoid(logits.values).cpu().detach().numpy()

        # Get prediction labels and boundary boxes
        labels = logits.indices.cpu().detach().numpy()
        boxes = outputs["pred_boxes"][0].cpu().detach().numpy()   

        return boxes, scores, labels


