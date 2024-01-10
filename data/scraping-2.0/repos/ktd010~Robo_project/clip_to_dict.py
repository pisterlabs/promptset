from PIL import Image
import requests
import torch

from transformers import CLIPProcessor, CLIPModel

import os
import asyncio
from openai import AsyncOpenAI
from dotenv import load_dotenv

import json

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

image = Image.open("dining.jpg")
labels=["cat", "dog","rabbit", "cow","plate","fork", "spoon"]
inputs = processor(text=labels, images=image, return_tensors="pt", padding=True)

outputs = model(**inputs)
logits_per_image = outputs.logits_per_image  # this is the image-text similarity score
print(logits_per_image)
probs = logits_per_image.softmax(dim=1)  # we can take the softmax to get the label probabilities

# Get the indices of the highest 3 probabilities
topk_values, topk_indices = torch.topk(probs, k=3, dim=1)

print("Top 3 indices:", topk_indices)

index_tensor = torch.tensor([[5, 4, 6]])

# Extracting labels using the provided indices
mapped_labels = [labels[index.item()] for index in index_tensor[0]]

print(mapped_labels)

#get labels of present objects

#mapped_labels = ['fork', 'plate', 'spoon']
prompt1 = f"In a dining table setting, from {', '.join(mapped_labels)}  assign 0 for items to be removed and assign 1 to keep. write objects as python dictionary. write this as  very brief "
#print(prompt1)

load_dotenv()

client = AsyncOpenAI(
    # This is the default and can be omitted
    api_key=os.environ.get("OPENAI_API_KEY"),
)


# Declare prompt2 as a global variable
prompt2 = {}

async def main() -> None:
    global prompt2

    chat_completion = await client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": prompt1,
            }
        ],
        model="gpt-3.5-turbo",
    )
    
    completion_content = chat_completion.choices[0].message.content
    prompt2 = json.loads(completion_content)
    print(prompt2)

# Run the main function
asyncio.run(main())

# Now, prompt2 is accessible globally with the updated content_dict
print(prompt2)

