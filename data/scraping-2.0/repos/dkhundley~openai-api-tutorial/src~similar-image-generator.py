# Importing the necessary Python libraries
import yaml
from io import BytesIO
from PIL import Image
from base64 import b64decode
import openai
import gradio as gr



## OPENAI CONNECTION
## ---------------------------------------------------------------------------------------------------------------------
# Loading the API key and organization ID from file (NOT pushed to GitHub)
with open('../keys/openai-keys.yaml') as f:
    keys_yaml = yaml.safe_load(f)

# Applying our API key and organization ID to OpenAI
openai.organization = keys_yaml['ORG_ID']
openai.api_key = keys_yaml['API_KEY']



## GRADIO HELPER FUNCTIONS
## ---------------------------------------------------------------------------------------------------------------------
def generate_similar_images(upload_image):
    '''
    Generates similar images based on an input image

    Inputs:
        - upload_image (PIL): An image uploaded by the user that will be the basis to create similar images

    Returns:
        - output_gallery (list): A list of images that will be returned in a display gallery
    '''
    # Using DALL-E to generate the image as a base64 encoded object
    openai_response = openai.Image.create_variation(
        image = open(upload_image, 'rb'),
        n = 5,
        size = '1024x1024',
        response_format = 'b64_json'
    )

    # Creating a list to hold all the images as the output gallery
    output_gallery = []

    # Iterating through all the images returned by DALL-E
    for image in openai_response['data']:
        
        # Appending the DALL-E generated image to the gallery
        output_gallery.append(Image.open(BytesIO(b64decode(image['b64_json']))))

    return output_gallery


## GRADIO UI LAYOUT & FUNCTIONALITY
## ---------------------------------------------------------------------------------------------------------------------
# Defining the building blocks that represent the form and function of the Gradio UI
with gr.Blocks(title = 'DALL-E Similar Image Generator', theme = 'base') as similar_image_generator:
    
    # Instantiating the UI interface
    header = gr.Markdown('''
    # DALL-E Similar Images Generator
    
    Upload an image of what you would like DALL-E to produce a gallery of similar images. Please note that this upload image must be a `.png` image and must be less than 4MB.
    '''
    )
    upload_image = gr.Image(label = 'Image Uploader', type = 'filepath')
    generate_similar_images_button = gr.Button('Generate Similar Images')
    output_gallery = gr.Gallery(label = 'Similar Image Gallery', object_fit = 'scale-down')
    examples = gr.Examples(
        examples = ['../data/car.png'],
        inputs = upload_image,
        outputs = output_gallery,
        fn = generate_similar_images
    )

    # Defining the behavior for when the "Generate Similar Images" button is clicked
    generate_similar_images_button.click(fn = generate_similar_images,
                                         inputs = [upload_image],
                                         outputs = [output_gallery])




## SCRIPT INVOCATION
## ---------------------------------------------------------------------------------------------------------------------
if __name__ == "__main__":

    # Launching the Gradio UI
    similar_image_generator.launch()