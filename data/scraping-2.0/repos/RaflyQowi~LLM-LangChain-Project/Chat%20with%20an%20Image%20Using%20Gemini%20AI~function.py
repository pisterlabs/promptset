from langchain.tools import BaseTool
from PIL import Image, ImageDraw
import requests
from dotenv import load_dotenv
import os
load_dotenv()


def object_detection_query(filepath):
    API_URL = "https://api-inference.huggingface.co/models/facebook/detr-resnet-50"
    headers = {"Authorization": "Bearer " + os.environ['HUGGINGFACEHUB_API_TOKEN']}
    with open(filepath, "rb") as f:
        data = f.read()
    response = requests.post(API_URL, headers=headers, data=data)
    return response.json()

def bounding_box(filepath):
    # Generate an output
    output = object_detection_query(filepath)

    # load the image
    image = Image.open(filepath).convert('RGB')

    # create a drawing object
    draw = ImageDraw.Draw(image)

    # Draw boxes and labels on the image
    for detection in output:
        label = detection['label']
        score = detection['score']
        box = detection['box']

        # Draw the box
        draw.rectangle([box['xmin'], box['ymin'], box['xmax'], box['ymax']], outline="red", width=2)

        # Draw the label and score
        text = f"{label} ({score:.2f})"
        draw.text((box['xmin'], box['ymin']-10), text, fill='red')

    return image

def captioning_query(filepath):
    API_URL = "https://api-inference.huggingface.co/models/Salesforce/blip-image-captioning-large"
    headers = {"Authorization": "Bearer " + os.environ['HUGGINGFACEHUB_API_TOKEN']}
    with open(filepath, "rb") as f:
        data = f.read()
    response = requests.post(API_URL, headers=headers, data=data)
    return response.json()

class ImageCaptionTools(BaseTool):
    name = "Image_Caption_Tools"
    description = "Use this tool with any given image path to receive a personalized description, poem, story, or more. "\
                  "Ideal for agents seeking tailored insights. "\
                  "Let the tool craft content based on your image for a unique perspective."

    def _run(self, image_path) -> str:
        """Use the tool."""
        result = captioning_query(image_path)
        text = result[0]['generated_text']
        return text

    async def _arun(self, query: str) -> str:
        """Use the tool asynchronously."""
        raise NotImplementedError("custom_search does not support async")
    

class ObjectDetectionTool(BaseTool):
    name = "Object_Detection_Tool"
    description = "Object Detection Tool: Use this tool to detect objects in an image. Provide the image path, " \
                  "and it will return a list of detected objects. Each element in the list is in the format: " \
                  "[x1, y1, x2, y2] class_name confidence_score. This tool focuses on object detection, providing " \
                  "locations of objects in the image. For image descriptions or other insights, explore additional tools."

    def _run(self, image_path) -> str:
        """Use the tool."""
        results = object_detection_query(image_path)
        detections = ""
        for result in results:
            box = result['box']
            detections += '[{}, {}, {}, {}]'.format(int(box['xmin']), int(box['ymin']), int(box['xmax']), int(box['ymax']))
            detections += ' {}'.format(result['label'])
            detections += ' {}\n'.format(result['score'])
        return detections

    async def _arun(self, query: str) -> str:
        """Use the tool asynchronously."""
        raise NotImplementedError("custom_search does not support async")

