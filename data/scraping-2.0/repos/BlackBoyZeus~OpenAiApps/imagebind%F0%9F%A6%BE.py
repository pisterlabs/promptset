#In this example, we first set up our OpenAI API credentials and the image bind model, and load our data into inputs. We then use the OpenAI API to generate a question about the images, which we add to our inputs. Finally, we generate embeddings using the image bind model and output the results.

#Note that this is just a simple example of how you can combine the OpenAI API with the image bind model, and there are many other ways you could use the API to enhance the functionality of the model.

import openai
import torch
from models import imagebind_model
from models.imagebind_model import ModalityType

# Set up OpenAI API credentials
openai.api_key = "YOUR_API_KEY"

# Set up image bind model
text_list=["A dog.", "A car", "A bird"]
image_paths=[".assets/dog_image.jpg", ".assets/car_image.jpg", ".assets/bird_image.jpg"]
audio_paths=[".assets/dog_audio.wav", ".assets/car_audio.wav", ".assets/bird_audio.wav"]

device = "cuda:0" if torch.cuda.is_available() else "cpu"

model = imagebind_model.imagebind_huge(pretrained=True)
model.eval()
model.to(device)

# Load data into inputs
inputs = {
    ModalityType.TEXT: data.load_and_transform_text(text_list, device),
    ModalityType.VISION: data.load_and_transform_vision_data(image_paths, device),
    ModalityType.AUDIO: data.load_and_transform_audio_data(audio_paths, device),
}

# Generate text using OpenAI API
prompt = "What is in the images?"
model_engine = "davinci"

response = openai.Completion.create(
  engine=model_engine,
  prompt=prompt,
  max_tokens=50,
  n=1,
  stop=None,
  temperature=0.5,
)

question = response.choices[0].text.strip()

# Add OpenAI API text to inputs
inputs[ModalityType.TEXT] = data.load_and_transform_text([question], device)

# Generate embeddings using image bind model
with torch.no_grad():
    embeddings = model(inputs)

# Output results
print(
    "Vision x Text: ",
    torch.softmax(embeddings[ModalityType.VISION] @ embeddings[ModalityType.TEXT].T, dim=-1),
)
print(
    "Audio x Text: ",
    torch.softmax(embeddings[ModalityType.AUDIO] @ embeddings[ModalityType.TEXT].T, dim=-1),
)
print(
    "Vision x Audio: ",
    torch.softmax(embeddings[ModalityType.VISION] @ embeddings[ModalityType.AUDIO].T, dim=-1),
)
