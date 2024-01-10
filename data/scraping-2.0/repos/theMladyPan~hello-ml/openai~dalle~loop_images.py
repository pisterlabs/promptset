import os
import openai
import dotenv
import sys 
import argparse
from PIL import Image
from io import BytesIO
import base64
import logging

log = logging.getLogger(__name__)

logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(message)s"
)

parser = argparse.ArgumentParser(description='Create N subsequent variations of image.')
parser.add_argument(
    'file',
    metavar='FILE',
    type=str,
    help='file to edit'
)
parser.add_argument(
    '-l',
    "--loop",
    metavar='LOOP',
    required=False,
    type=int,
    default=3,
    help='number of times to loop, default is 1'
)
parser.add_argument(
    '-s',
    "--size",
    metavar='SIZE',
    required=False,
    type=str,
    default="256x256",
    help='size of images to generate, default is 256x256'
)

args = parser.parse_args()

# Load the environment variables
dotenv.load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
log.info("Loaded OpenAI API key.")

filename = args.file
log.info(f"Starting with file {filename}.")

for i in range(args.loop):
    # generate image from text prompt
    log.info(f"Creating variation {i+1} of {filename}.")
    response = openai.Image.create_variation(
        image=open(filename, "rb"),
        n=1,
        size=args.size,
        response_format="b64_json"
    )

    filename = f"images/loop_{i+1}.png"
    log.info(f"Saving image to {filename}.")
    #save image to file
    data = response["data"]
    img = Image.open(BytesIO(base64.b64decode(data[0]["b64_json"])))
    img.save(filename)
