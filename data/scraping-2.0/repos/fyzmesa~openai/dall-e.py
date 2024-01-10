import os
import openai
import base64
from datetime import datetime
from IPython.display import Image
from IPython import display
from base64 import b64decode

openai.api_key = ""

##############################################################################

prompt = "A futuristic neon lit cyborg face"

##############################################################################

i = 1

while i < 4:
    now = datetime.now()
    timestamp = now.strftime("%Y%m%d%H%M%S")

    response = openai.Image.create(
        prompt=prompt,
        model="image-alpha-001",
        size="512x512",
        response_format="b64_json"
        )

    image_b64 = response['data'][0]['b64_json']

    imgdata = base64.b64decode(image_b64)

    filename = 'outputs/%simage.jpg' %timestamp

    with open(filename, 'wb') as f:
        f.write(imgdata)

    i += 1
    
# PROMPTS USED ###############################################################

# a realistic photography of a crocodile wearing Louix XVI style outfits
# a realistic photography of a panda wearing Samouraï style outfits
# a realistic photography of a panda wearing Samouraï armors
# a hand drawing of Mickey Mouse playing piano in a beautiful old fashion restaurant
# a black and white hand drawing of Mickey Mouse playing piano in a beautiful old fashion restaurant
# a realistic photography of a crocodile smiling and wearing cosmonauts outfits
# a realistic photography of a female ballet dancer who's skin is made out of silver
# a geometrical and symmetrical drawing of a marmote
# a realistic photography of a young beautiful japanese woman
# a banana wearing a suit standing inside an elevator
# A futuristic neon lit cyborg face
# A sea otter with a pearl earring by Johannes Vermeer
# A photograph of a sunflower with sunglasses on in the middle of the flower in a field on a bright sunny day
