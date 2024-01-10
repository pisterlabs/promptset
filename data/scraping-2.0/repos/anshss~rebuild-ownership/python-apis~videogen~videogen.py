#pip install replicate
import replicate
import os
from dotenv import load_dotenv
from openai import OpenAI
import boto3
from io import BytesIO
from urllib.parse import quote, urlparse
from pydub import AudioSegment
#pip install ibm_watson
from ibm_watson import TextToSpeechV1
from ibm_cloud_sdk_core.authenticators import IAMAuthenticator

load_dotenv()
api_key = os.environ.get("OPENAI_API_KEY")
apikey=os.environ.get("watson_apikey")
url=os.environ.get("watson_url")

client = OpenAI()

#setup ibm service

#Setup Service
authenticator = IAMAuthenticator(apikey)
#New TTS Service
tts= TextToSpeechV1(authenticator=authenticator)
#Set Service URL
tts.set_service_url(url)

load_dotenv()

s3 = boto3.client(
    's3',
    aws_access_key_id=os.environ.get("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=os.environ.get("AWS_SECRET_ACCESS_KEY"),
   
)

s3_bucket_name="bucketforadgen"

def url_to_uri(url):
    parsed_url = urlparse(url)
    uri = parsed_url.scheme + "://" + parsed_url.netloc + quote(parsed_url.path)
    
    if parsed_url.query:
        uri += quote("?" + parsed_url.query)
    
    if parsed_url.fragment:
        uri += quote("#" + parsed_url.fragment)
    
    return uri

def videogen(model_image, product_name, product_description, model_gender):
    model_voice = audiogen(product_name, product_description, model_gender)
    model_image_encoded = url_to_uri(model_image)
    print(f"Model Image URI: {model_image_encoded}")

    output = replicate.run(
        "cjwbw/sadtalker:3aa3dac9353cc4d6bd62a8f95957bd844003b401ca4e4a9b33baa574c549d376",
        input={
            "still": True,
            "enhancer": "gfpgan",
            "preprocess": "full",
            "driven_audio": model_voice,
            "source_image": model_image_encoded
        }
    )
    
    try:
        print(output)
    except Exception as e:
        print(f"Error uploading video to S3: {e}")



def audiogen(product_name, product_description, model_gender):
    script = generate_script(product_name, product_description)

    try:
        s3_audio = None
        if model_gender == "female":
            with open('./generated-audio-female.mp3', 'wb') as audio_file:
                response = tts.synthesize(script, accept='audio/mp3', voice='en-US_AllisonV3Voice').get_result()
                generated_audio = response.content
                audio_file.write(generated_audio)
        elif model_gender == "male":
            with open('./generated-audio-male.mp3', 'wb') as audio_file:
                response = tts.synthesize(script, accept='audio/mp3', voice='en-US_HenryV3Voice').get_result()
                generated_audio = response.content
                if response.status_code != 200:
                    print(f"Error synthesizing audio. Status code: {response.status_code}")
                audio_file.write(generated_audio)
        

          # Convert the generated audio from mp3 to wav
        audio = AudioSegment.from_mp3(f'./generated-audio-{model_gender}.mp3')
        audio.export(f'./generated-audio-{model_gender}.wav', format='wav')
        # Read the converted wav
        with open(f'./generated-audio-{model_gender}.wav', 'rb') as wav_file:
            audio_bytes = BytesIO(wav_file.read())

        
        
        s3_audio = upload_audio_to_s3(audio_bytes, product_name)
    except Exception as e:
        print(f"Error generating or uploading audio: {e}")

    return s3_audio


def upload_audio_to_s3(audio_bytes, product_name):
    try:
        product_name_cleaned = product_name.replace(" ", "_")
        s3_bucket_name = 'bucketforadgen'
        s3_key = f"{product_name_cleaned}_generated_audio.mp3"

        # Upload the audio to S3
        s3.put_object(Body=audio_bytes, Bucket=s3_bucket_name, Key=s3_key, ContentType='audio/mpeg')

        audio_public_url = f'https://{s3_bucket_name}.s3.amazonaws.com/{s3_key}'
        
        return audio_public_url
    except Exception as e:
        print(f"Error uploading audio to S3: {e}")
        raise e

def generate_script(product_name, product_description):
    script_prompt = f"Create a short catchy advertisement script for a product named {product_name}. Description: {product_description}"
    script_response = client.completions.create(
        model="text-davinci-003",
        prompt=script_prompt,
        max_tokens=50
    )
    
    script = script_response.choices[0].text.strip().replace('"', '')
    script = [line.replace('\n', '') for line in script]
    script = ''.join(str(line) for line in script) 
   
    print(f"Script: {script}")
    
    return script

if __name__ == "__main__":
    product_name = "laptop"
    product_description = "apple's new gen m5 chip laptop"
    model_img = "https://bucketforadgen.s3.ap-south-1.amazonaws.com/macbook_air_ad_poster.png"
    model_gender = "female"
    videogen(model_img, product_name, product_description, model_gender)
