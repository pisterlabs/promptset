import os
import cv2
import inference
import torch
import supervision as sv
from openai import OpenAI

client = OpenAI()

# Replace with your OpenAI API key
openai_api_key = os.getenv('OPENAI_API_KEY')
annotator = sv.BoxAnnotator()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Function to read instructions/goals from a text file
def read_instructions_from_file(file_path):
    with open(file_path, 'r') as file:
        instructions = file.read()
    return instructions

# Create an OpenAI GPT-4 conversation with the instructions
def create_gpt4_conversation(instructions):
    conversation = [
        {"role": "system", "content": "You are a going to interact with a webcam and use that info of playing cards there to help a player win at blackjack"},
        {"role": "user", "content": instructions},
    ]
    return conversation

# Function to handle OpenAI GPT-4 responses
def handle_gpt4_response(response):
    # Print the entire response for debugging

    # Accessing the content of the response
    if 'choices' in response and len(response['choices']) > 0:
        model_response = response['choices'][0]['message']['content']
        print("GPT-4 Response:", model_response)
    else:
        print("")


def on_prediction(predictions, image):
    labels = [p["class"] for p in predictions["predictions"]]
    detections = sv.Detections.from_roboflow(predictions)
    detections = detections[detections.confidence > 0.45]
    print(detections)
    
    # Read instructions from the text file "Player.txt"
    instructions = read_instructions_from_file("Player.txt")
    
    # Create a GPT-4-turbo conversation with the instructions
    conversation = create_gpt4_conversation(instructions)
    
    # Use GPT-4-turbo to get a response
    gpt4_response = client.chat.completions.create(model="gpt-4-1106-preview",
    messages=conversation)
    
    # Print the structure of the GPT-4 response (for debugging)
    #print(gpt4_response)

    # Handle the GPT-4 response
    handle_gpt4_response(gpt4_response)
    
    # Annotate and display the image with predictions
    annotated_image = annotator.annotate(
        scene=image, 
        detections=detections,
        labels=labels
    )
    cv2.imshow("Prediction", annotated_image)
    cv2.waitKey(1)

# Initialize and run the inference stream
inference.Stream(
    source="webcam",  # Use the default webcam
    model="black-jack-8goxw/13",  # ML model from Universe
    output_channel_order="BGR",  # Color channel order for OpenCV
    use_main_thread=True,  # for OpenCV display
    on_prediction=on_prediction,
    api_key=''  # Replace with your actual API key if required
)

# Cleanup: Release the webcam and close OpenCV window
cv2.destroyAllWindows()
