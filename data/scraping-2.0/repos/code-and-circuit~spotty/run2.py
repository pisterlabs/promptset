import os
import sys
import uuid
import pathlib
import requests
import json
import openai
import re
import cv2

from elevenlabs import set_api_key
from elevenlabs import generate, play
eleven_key = "df9fff70db36babd0e5040610c8c822f"
set_api_key(eleven_key)


filename = "temp.png" #sys.argv[1]

cam_port = 0
cam = cv2.VideoCapture(cam_port)

result, image = cam.read()

if result:
    # saving image in local storage
    cv2.imwrite("temp.png", image)

file_extension = pathlib.Path(filename).suffix
uid = str(uuid.uuid4())
file_path = filename
uname = uid + file_extension

account_name = 'codeandcircuituploads'
account_key = 'IVYI0Q+T4LBtjBMjK4WGJB/bgRBZNxvVRZdlFnvWR2gyqS7oX9P9Q8JgYKagYAUteUmxTS7y9iIl+ASti+61kg=='
container_name = 'images'

account_url = "https://" + account_name + ".blob.core.windows.net"

from datetime import datetime, timedelta
from azure.storage.blob import BlobServiceClient, generate_account_sas, ResourceTypes, AccountSasPermissions

sas_token = generate_account_sas(
    account_name=account_name,
    account_key=account_key,
    resource_types=ResourceTypes(service=True, container=True, object=True),
    permission=AccountSasPermissions(read=True, write=True),
    expiry=datetime.utcnow() + timedelta(hours=1)
)

blob_service_client = BlobServiceClient(account_url=account_url, credential=sas_token)

# Create a blob client using the local file name as the name for the blob
blob_client = blob_service_client.get_blob_client(container=container_name, blob=uname)

with open(file=filename, mode="rb") as data:
    blob_client.upload_blob(data=data)

#block_blob_service = BlockBlobService( account_name=account_name, account_key=account_key)
#block_blob_service.create_blob_from_path(container_name, uname, file_path)

cloud_url = "https://" + account_name + ".blob.core.windows.net/" + container_name + "/" + uname
print(cloud_url)
api_url = "https://cctest1.cognitiveservices.azure.com/computervision/imageanalysis:analyze?api-version=2023-02-01-preview&features=denseCaptions&language=en&gender-neutral-caption=False"
post = {"url" : cloud_url}
headers = {"Ocp-apim-subscription-key":"243a6fc6116940849c58b4eb168ebdfa" , "Content-Type":"application/json"}
response = requests.post(api_url, json=post, headers=headers)

captions = json.loads(response.text)

print(captions)
print(response.status_code)

prompt = "Pretend you are a curious child that is observing the world. You see the following scene: \n"

count = 1
for thing in captions["denseCaptionsResult"]["values"]:
  prompt += str(count) + ": " + thing["text"] + "\n"
  count = count + 1

print(prompt)
  

prompt += "Pick one item, and respond with a JSON response about why that item is interesting to you. Also include in the JSON one sentence you'll say when you see it, and another sentence you'd say when you get there. The keys of the JSON respnose should be index, description, sentence1, and sentence2"

openai.api_key = "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";

completion = openai.ChatCompletion.create(
  model="gpt-3.5-turbo",
  messages=[
    {"role": "user", "content": prompt}
  ]
)

content = completion.choices[0].message.content
print(content)

regex = r"\{(.*?)\}"

matches = re.finditer(regex, content, re.MULTILINE | re.DOTALL)

for matchNum, match in enumerate(matches):
    for groupNum in range(0, len(match.groups())):
        content = (match.group(1))

content = "{" + content + "}"

print(content)

result = json.loads(content)

def say(result, sentence):

  if sentence in result:
 
    greeting = result[sentence]

    audio = generate(
    text=greeting,
    voice="Sam",
    model="eleven_monolingual_v1"
    )  

    play(audio)

say(result, "sentence1")

if "index" in result:

  item = captions["denseCaptionsResult"]["values"][result["index"]-1]
  box = item["boundingBox"]
  horiz = box["x"] + (box["w"]/2)
  vert = box["y"] + (box["h"]/2)

  print("Turn to point at " + str(horiz) + ", " + str(vert) )

print("walking...")

say(result, "sentence2")

