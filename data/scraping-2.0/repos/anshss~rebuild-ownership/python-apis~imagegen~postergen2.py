# Importing Dependencies
from openai import OpenAI
from PIL import Image
from io import BytesIO
import requests
from dotenv import load_dotenv
import os
import boto3

# Importing API key from .env file
load_dotenv()
api_key = os.environ.get("OPENAI_API_KEY")
removebg_api_key = os.environ.get("REMOVEBG_API_KEY")

# Initialize AWS S3 client
s3 = boto3.client(
    's3',
    aws_access_key_id=os.environ.get("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=os.environ.get("AWS_SECRET_ACCESS_KEY"),
    
)
region=os.environ.get("AWS_DEFAULT_REGION")  # Add region
# Set S3 bucket name (use a constant)
S3_BUCKET_NAME = 'bucketforadgen'

# Initialize OpenAI client
client = OpenAI()



# Function to upload image to S3
def upload_to_s3(image, product_name):
    try:
        # Replace spaces with underscores in the product name
        product_name_cleaned = product_name.replace(" ", "_")

        # Set S3 key based on cleaned product name
        s3_key = f"{product_name_cleaned}_ad_poster.png"

        # Create a BytesIO object to hold the image content
        image_bytesio = BytesIO()
        image.save(image_bytesio, format='PNG')
        image_bytes = image_bytesio.getvalue()

        # Upload the image to S3
        s3.put_object(Body=image_bytes, Bucket=S3_BUCKET_NAME, Key=s3_key,ContentType="image/png")

        # Get the public URL for the uploaded image
        s3_public_url = f'https://{S3_BUCKET_NAME}.s3.amazonaws.com/{s3_key}'
        print(f"Ad poster uploaded to S3. Public URL: {s3_public_url}")
        return s3_public_url

    except Exception as e:
        print(f"Error uploading to S3: {e}")

        


# Function to generate ad poster with two images
def generate_ad_poster_with_images(product_name, product_description, image_url1, image_url2):
    try:
        # Remove background using remove.bg for the first image
        removebg_url = "https://api.remove.bg/v1.0/removebg"
        headers = {"X-Api-Key": removebg_api_key}

        # Make a request to remove.bg API for the first image
        removebg_response1 = requests.post(removebg_url, headers=headers, data={'image_url': image_url1, 'size': 'auto'})
        
        # Check if the remove.bg API call was successful for the first image
        if removebg_response1.status_code == 200:
            # Convert the response content to bytes
            removebg_content1 = BytesIO(removebg_response1.content)

            # Open the first image with transparency using PIL
            image1_without_bg = Image.open(removebg_content1).convert("RGBA")

            # Remove background using remove.bg for the second image
            removebg_response2 = requests.post(removebg_url, headers=headers, data={'image_url': image_url2, 'size': 'auto'})
            
            # Check if the remove.bg API call was successful for the second image
            if removebg_response2.status_code == 200:
                # Convert the response content to bytes
                removebg_content2 = BytesIO(removebg_response2.content)

                # Open the second image with transparency using PIL
                image2_without_bg = Image.open(removebg_content2).convert("RGBA")

                # Generate a catchy tagline using OpenAI API
                tagline_prompt = f"Create a short catchy tagline for a product named {product_name}. Description: {product_description}"
                tagline_response = client.completions.create(
                    model="text-davinci-003",
                    prompt=tagline_prompt,
                    max_tokens=50
                )
                tagline = tagline_response.choices[0].text.strip().replace('"', '')

                # Generate an ad poster with the two images and tagline
                dalle_api_prompt = f"Generate a solid background image which focuses on top and contains only tagline  should be placed at the bottom only, in which we can paste the image later on top of the generated background.  Tagline: {tagline}"
                dalle_response = client.images.generate(
                    model="dall-e-3",
                    prompt=dalle_api_prompt,
                    size="1024x1024",
                    quality="hd",
                    n=1,
                )
#in which we can paste the images of a {product_name}
                # Download the generated image
                generated_image_content = BytesIO(requests.get(dalle_response.data[0].url).content)

                # Open the generated image using PIL
                generated_image = Image.open(generated_image_content).convert("RGBA")

                # Paste the first image onto the ad poster
                generated_image.paste(image1_without_bg, (50, 50), mask=image1_without_bg)

                # Paste the second image onto the ad poster
                generated_image.paste(image2_without_bg, (200, 200), mask=image2_without_bg)

                # Upload the ad poster to S3
                s3_public_url = upload_to_s3(generated_image, product_name)
                

            else:
                print(f"Error removing background for the second image. Status Code: {removebg_response2.status_code}")
            
        else:
            print(f"Error removing background for the first image. Status Code: {removebg_response1.status_code}")
            
    except Exception as e:
        print(f"Error generating ad poster: {e}")


# Example usage
if __name__ == "__main__":
    product_name = "macbook air"
    product_description = "apple's latest macbook with the latest m5 chip"
    image_url1 = r"https://bucketforadgen.s3.ap-south-1.amazonaws.com/realistic_professional_girl_like_real_world_model_img.png"
    image_url2 = r"https://bucketforadgen.s3.ap-south-1.amazonaws.com/macbook-air-laptop-11549346117llrtyx80ne.png"
    generate_ad_poster_with_images(product_name, product_description, image_url1, image_url2)
