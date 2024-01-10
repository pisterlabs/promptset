import openai
import open_clip
import torch
from PIL import Image
import cv2
import json

# Load the OpenAI API key
openai.api_key = ""
# Load the CLIP model
model, _, transform = open_clip.create_model_and_transforms(
    model_name="coca_ViT-L-14",
    pretrained="mscoco_finetuned_laion2B-s13B-b90k"
)

# Define the video path
video_path = "/Users/jimvincentwagner/tests/video_1680602199_RFREfQhFxQ.mp4"

# Open the video capture
cap = cv2.VideoCapture(video_path)

# Get the total number of frames
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# Create an empty list to store the output captions
captions = []

# Loop through every 5th frame of the video
for i in range(0, total_frames, 5):
    # Set the video capture to the current frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, i)

    # Read the frame from the video capture
    ret, frame = cap.read()

    # Check if there are no more frames
    if not ret:
        break

    # Convert the frame to RGB and apply the transformation
    im = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    im = transform(im).unsqueeze(0)

    # Run inference on the frame
    with torch.no_grad(), torch.cuda.amp.autocast():
        generated = model.generate(im)

    # Decode the output caption
    caption = open_clip.decode(generated[0]).split("<end_of_text>")[0].replace("<start_of_text>", "")

    # Append the caption to the list
    captions.append(caption)

    # Print the progress
    print(f"Processed frame {i+1} of {total_frames}")

# Release the video capture
cap.release()

# Create a dictionary to store the captions
output_dict = {"captions": captions}

# Write the output to a JSON file
with open("output.json", "w") as f:
    json.dump(output_dict, f)

# Load the captions from the JSON file
with open("output.json", "r") as f:
    captions = json.load(f)["captions"]

# Combine the captions into a single string
prompt_lines = "\n".join(captions)

# Set up OpenAI Chat Completion API parameters
MODEL = "gpt-3.5-turbo-0301"

response = openai.ChatCompletion.create(
    model=MODEL,
    messages=[
        {"role": "system", "content": "You are aiding in categorizing Videos." },
        {"role": "user", "content": "The following Lines are the captions that Computer Vision Information Retrieval model outputs. The Intention is to categorize and label the video. Provide 5 hashtags for it. Just the regular ones of a social media app. Also Place it into one of the categories: 1. Sports , 2. User-generated content, Private Event, 5. Outside with people, 6. inside of the appartment . Explain."},
        {"role": "assistant", "content": "Okay, so what are the captions?"},
        {"role": "user", "content": "Captions:\n\n" + prompt_lines},
        {"role": "system", "content": "Now filter everything tha is not a hashtag and more likely just because the vision model just randomly picked up on it."},
    ],
    temperature=0,
)

print(response.choices[0]["message"]["content"])
