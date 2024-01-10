#!/usr/bin/env python3
import requests
import datetime
import openai
from google.cloud import storage
from ayrshare import SocialPost


social = SocialPost("AYRSHARE_API_KEY")
openai.api_key ="OPENAI_API_KEY"
CurrentDate=datetime.datetime.now().strftime("%d%m%Y")


#OpenAI API to generete a story about a random topic
responseS = openai.Completion.create(
  model="text-davinci-002",
  prompt="tell me a story about a random topic",
  temperature=0.7,
  max_tokens=256,
  top_p=1,
  frequency_penalty=0,
  presence_penalty=0
)
#formatting the story string
StoryString = responseS['choices'][0]['text']
StoryString = StoryString.strip('\n')
StoryString= str(StoryString)
story=StoryString.replace('"', '')
story=story.replace("'", "")


#Dall-e API generate image according to the story
responseI = openai.Image.create(
  prompt= story,
  n=1,
  size="1024x1024"
)
ImgUrl = responseI['data'][0]['url']
img_data = requests.get(ImgUrl).content

#save the image locally
with open("img/image_"+CurrentDate+".jpg", "wb") as handler:
    handler.write(img_data)


#upload the image to the Google cloud storage bucket
source_file_name="img/image_"+CurrentDate+".jpg"
bucket_name="BUCKET_NAME"
destination_blob_name="img/image_"+CurrentDate+".jpg"
storage_client = storage.Client()
bucket = storage_client.bucket(bucket_name)
blob = bucket.blob(destination_blob_name)
blob.upload_from_filename(source_file_name)


#Ayrshare API post image and story to instagram
postResult = social.post({'post': StoryString, 'platforms': ['instagram'], 'mediaUrls': ['https://storage.googleapis.com/BUCKET_NAME/'+destination_blob_name]})
print(postResult)
