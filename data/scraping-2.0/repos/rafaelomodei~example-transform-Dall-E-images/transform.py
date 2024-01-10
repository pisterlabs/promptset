# Imports
import openai
import os, sys, shutil
import requests
import argparse

if not os.getenv('OPENAI_API_KEY'):
    print('Missing OPENAI_API_KEY')
    sys.exit(1)

if not os.getenv('EI_PROJECT_API_KEY'):
    print('Missing EI_PROJECT_API_KEY')
    sys.exit(1)

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
API_KEY = os.environ.get("EI_PROJECT_API_KEY")

# these are the three arguments that we get in
parser = argparse.ArgumentParser(description='Use OpenAI Dall-E to generate an image dataset for classification from your prompt')
parser.add_argument('--prompt', type=str, required=True, help="Prompt to give Dall-E to generate the images")
parser.add_argument('--label', type=str, required=True, help="Label for the images")
parser.add_argument('--images', type=int, required=True, help="Number of images to generate")
parser.add_argument('--variations', type=int, required=False,default=0, help="Number of variations per image")
parser.add_argument('--size', type=str, required=False, default="256x256", help="Image size in format 256x256")
parser.add_argument('--skip-upload', type=bool, required=False, help="Skip uploading to EI", default=False)
parser.add_argument('--out-directory', type=str, required=False, help="Directory to save images to", default="output")
args, unknown = parser.parse_known_args()
if not os.path.exists(args.out_directory):
    os.makedirs(args.out_directory)
output_folder = args.out_directory
    
# Set up OpenAI API key and organization
openai.api_key = OPENAI_API_KEY


prompt = args.prompt
label = args.label
image_size = args.size
base_images_number = args.images
variation_per_image = 0

if args.variations:
    variation_per_image = args.variations
output_folder = 'output/'
# Check if output directory exists and create it if it doesn't
if not os.path.exists(output_folder):
    os.makedirs(output_folder)
else:
    shutil.rmtree(output_folder)
    os.makedirs(output_folder)



for i in range(base_images_number):
    print(f'Creating image {i+1} of {base_images_number} for {label}...')
    try:
        response = openai.Image.create(
            prompt=prompt,
            n=1,
            size=image_size,
        )        
        img = response["data"][0]
        fullpath = os.path.join(args.out_directory,f'{label}.{i}.png')
        with open(fullpath, 'wb+') as f:
            png = requests.get(img["url"]).content 
            f.write(png)
        if variation_per_image > 0:
            print(f'Creating {variation_per_image} variations...')
            try:
                response2 = openai.Image.create_variation(
                    image=requests.get(img['url']).content,
                    n=variation_per_image,
                    size=image_size
                )
                n=1
                for img in response2['data']:
                    fullpath = os.path.join(args.out_directory,f'{label}.{i}_{n}.png')
                    with open(fullpath, 'wb+') as f:
                        png = requests.get(img["url"]).content 
                        f.write(png)
                    n +=1
            except Exception as e:
                print('Failed to complete DALL-E generation:', e)
    except Exception as e:
        print('Failed to complete DALL-E generation:', e)
 
    



if args.skip_upload:
    print('Skipping upload to Edge Impulse')
    sys.exit(0) 
headers = {
    'x-api-key': API_KEY,
    'x-label': label,
    'x-disallow-duplicates': 'true',
    'Content-Type': 'image/png'
}

# Iterate through the sub-directories in the given directory

for file in os.listdir(output_folder):
    file_path = os.path.join(output_folder, file)
    if os.path.isfile(file_path):
        with open(file_path, 'r') as file:
            res = requests.post(url='https://ingestion.edgeimpulse.com/api/training/files',
            headers={
            'x-label': label,
            'x-api-key': API_KEY,},
            files = { 'data': (os.path.basename(file_path), open(file_path, 'rb'), 'image/png') }
        )
    if (res.status_code == 200):
        print('Uploaded file to Edge Impulse', res.status_code, res.content)
    else:
        print('Failed to upload file to Edge Impulse', res.status_code, res.content)