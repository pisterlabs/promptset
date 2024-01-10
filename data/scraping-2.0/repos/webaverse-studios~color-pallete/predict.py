# Prediction interface for Cog ⚙️
# https://github.com/replicate/cog/blob/main/docs/python.md

from cog import BasePredictor, Input, Path
from typing import Any
import os

from PIL import Image
import openai
import ast
from io import BytesIO
import base64
from multiprocessing import set_start_method

from recolor import transfer, util, palette

try:
    set_start_method('spawn')
except RuntimeError:
    pass

print("openai key:", os.getenv("OPENAI_KEY", None))
            
def remap_colors(rgb_channels, rgb_values, num_colors):
    image_lab = transfer.rgb2lab(rgb_channels)
    mypalette = palette.build_palette(image_lab, num_colors)
    new_color_image = Image.new(mode="RGB", size=(1,1))
    new_color_image.putpixel( (0, 0), rgb_values )
    new_lab_image = transfer.rgb2lab(new_color_image)
    lab_color =new_lab_image.getpixel((0,0))
    mypalette.append((0,128,128))
    new_palette = mypalette.copy()
    new_palette[0] = lab_color
    image_lab_m = transfer.image_transfer(image_lab, mypalette, new_palette, sample_level=10, luminance_flag=0)
    # Get pixel data from LAB images
    image1_data = list(image_lab.getdata())
    image2_data = list(image_lab_m.getdata())

# Calculate the average of LAB values for each pixel
    averaged_data = []
    for lab1, lab2 in zip(image1_data, image2_data):
        average_lab = tuple(int((l1 + l2) / 2) for l1, l2 in zip(lab1, lab2))
        averaged_data.append(average_lab)

    # Create a new image with averaged LAB values
    averaged_image_lab = Image.new('LAB', image_lab.size)
    averaged_image_lab.putdata(averaged_data)

        
    return util.lab2rgb(averaged_image_lab)

def getRGB(object, adjective):
    outtext = openai.Completion.create(
        model="davinci",
        prompt="* the main color of grass in a chocolate world is light brown.\n* the main color of rocks in a lemon world is yellow.\n* the main color of "+object+" in a "+adjective+ " world is ",
        max_tokens=256,
        temperature=0,
        stop=['\n','.']
        )
    response = outtext.choices[0].text
    print('generated color')
    print(response)
    outtext = openai.Completion.create(
        model="davinci",
        prompt="* color: red rgb: (255,0,0)\n* color: blue rgb: (0,0,255)\n* color: "+response+" rgb: ",
        max_tokens=256,
        temperature=0,
        stop=['\n','*']
        )
    response = outtext.choices[0].text
    return response


def img2b4(img: Image, format = 'PNG'):
    im_file = BytesIO()
    img.save(im_file, format=format)
    im_bytes = im_file.getvalue()  
    im_b64 = base64.b64encode(im_bytes).decode('utf-8')
    return f'data:image/{format.lower()};base64,{im_b64}'   

class Predictor(BasePredictor):
    def setup(self):
        """Load the model into memory to make running multiple predictions efficient"""
        # self.model = torch.load("./weights.pth")

    def predict(
        self,
        object: str = Input(description="Object", default="sand"),
        adjective: str = Input(description="Object", default="sand"),
        image: Path = Input(description="Grayscale input image"),
    ) -> Any:
        """Run a single prediction on the model"""

        try:
            openai.api_key = os.getenv("OPENAI_KEY", None)
            rgb_image = Image.open(image)
            generated_color = getRGB(object, adjective)
            remapped = remap_colors(rgb_image, ast.literal_eval(generated_color.strip()), 2)
            res = dict()
            res['remapped'] = img2b4(remapped)
            return res
        except Exception as e:
            return f"Error: {e}"
