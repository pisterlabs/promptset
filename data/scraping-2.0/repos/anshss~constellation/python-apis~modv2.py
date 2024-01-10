from openai import OpenAI          
from flask_cors import CORS
import requests
import os
import boto3
from dotenv import load_dotenv
from io import BytesIO
from PIL import Image


load_dotenv()

s3 = boto3.client(
    's3',
    aws_access_key_id=os.environ.get("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=os.environ.get("AWS_SECRET_ACCESS_KEY"),
   
)



client = OpenAI()

def generate_model_img(model_description):
    try:
        
        dalle_api_prompt = f"Generate a realistic image of a model captured with a Nikon D850 and a Nikon AF-S NIKKOR 70-200mm f/2.8E FL ED VR lens, lit with high-key lighting to create a soft and ethereal feel, with a shallow depth of field --ar 2:3- with the following attributes: {model_description}"
        dalle_response = client.images.generate(
            model="dall-e-3",
            prompt=dalle_api_prompt,
            size="1024x1024",
            quality="hd",
            n=1,
        )

       
        image_content = BytesIO(requests.get(dalle_response.data[0].url).content)

        
        upload_to_s3(image_content, model_description)

    except Exception as e:
        print(f"Error: {e}")


def upload_to_s3(image_content, model_description):
    try:
        
        model_description_cleaned = model_description.replace(" ", "_")

        s3_bucket_name = 'bucketforadgen'
        
        s3_key = f"{model_description_cleaned}_model_img.png"

        
        s3.put_object(Body=image_content.getvalue(), Bucket=s3_bucket_name, Key=s3_key,ContentType='image/png')

        
        s3_public_url = f'https://{s3_bucket_name }.s3.amazonaws.com/{s3_key}'
        print(f"Public URL for the image: {s3_public_url}")
        return s3_public_url
    except Exception as e:
        print(f"Error uploading to S3: {e}")
        raise e
if __name__ == "__main__":
    model_description = "realistic "
    generate_model_img(model_description)