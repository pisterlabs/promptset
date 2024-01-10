# Imports the Google Cloud client library
from google.cloud import vision
import os
import openai
import fnmatch
import pandas as pd



api_key_file_path = 'GPTAPI.txt'

# directory_path for images
directory_path = 'rig/'



def GetLabels(path) -> vision.EntityAnnotation:
    """Provides a quick start example for Cloud Vision."""

    client = vision.ImageAnnotatorClient()

    with open(path, "rb") as image_file:
        content = image_file.read()

    image = vision.Image(content=content)

    response = client.label_detection(image=image)
    labels = response.label_annotations
    print(path)
    description = []
    for label in labels:
        print(label.description)
        description.append(label.description)

    if response.error.message:
        raise Exception(
            "{}\nFor more info on error messages, check: "
            "https://cloud.google.com/apis/design/errors".format(response.error.message)
        )
    
    return description




def get_image_paths(directory_path):
    """
    Get paths of all image files in the specified directory and its subdirectories.
    """
    image_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp']
    image_paths = []
    
    for root, _, files in os.walk(directory_path):
        for file in files:
            if any(file.lower().endswith(ext) for ext in image_extensions):
                image_paths.append(os.path.join(root, file))
                
    return image_paths

#Find image paths
image_paths = get_image_paths(directory_path)

alllabels = []
for path in image_paths:
    labels = GetLabels(path)
    alllabels.extend(labels)  # Now storing labels in all_labels list




# Read the API key from the file
with open(api_key_file_path, 'r') as file:
    openai.api_key = file.read().strip()

def extract_building_materials(all_labels):
    # Combine all the labels into a single text string
    labels_text = ', '.join(set(all_labels))  # Using a set to remove duplicates

    # Construct a conversation with a user message and system message
    conversation_history = [
        {"role": "system", "content": "You are a helpful assistant that extracts building material keywords."},
        {"role": "user", "content": f"The following are labels describing building materials in images: {labels_text}. Extract and list only the keywords that are specifically related to building materials."}
    ]

    # Send the conversation to the Chat API
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",  # You can choose other models as well
        messages=conversation_history,
        max_tokens=100  # Adjust as needed
    )

    # Extract the assistant's reply from the response
    assistant_reply = response.choices[0].message['content'].strip()

    # Assuming the model returns a comma-separated list, split into individual keywords
    # You may need to adjust this depending on how your model structures the response
    keywords = [keyword.strip() for keyword in assistant_reply.split(',')]

    return keywords

# Example usage
building_material_keywords = extract_building_materials(alllabels)

# Do something with the extracted keywords
print(building_material_keywords)

# # Write the keywords to a file
# with open('keywords.txt', 'w') as file:
#     file.write('\n'.join(building_material_keywords))

# # write alllabels to a file
# with open('alllabels.txt', 'w') as file:
#     file.write('\n'.join(alllabels))



# conversation_history = [
#     {"role": "system", "content": "You are an assistant knowledgeable in material sustainability."},
#     {"role": "user", "content": f"Please rate the following materials based on the ability to downcycle, upcycle, recycle, repurpose, or landfill: {', '.join(building_material_keywords)}. Give a rating from 1 to 3 for each material, with 1 being the best and 3 being the worst."}
# ]

# response = openai.ChatCompletion.create(
#     model="gpt-3.5-turbo",  # Update to the model you are using
#     messages=conversation_history,
#     max_tokens=1000  # Increase if needed to ensure complete responses
# )

# assistant_reply = response.choices[0].message['content'].strip()

### to debug images and labels
# from IPython.display import display, Image
# import cv2
# import io

# def GetDisplayLabels(path):
#     """Provides a quick start example for Cloud Vision."""
#     client = vision.ImageAnnotatorClient()

#     with open(path, "rb") as image_file:
#         content = image_file.read()

#     image = vision.Image(content=content)

#     response = client.label_detection(image=image)
#     labels = response.label_annotations
#     print(path)
#     description = []
#     for label in labels:
#         print(label.description)
#         description.append(label.description)

#     if response.error.message:
#         raise Exception(
#             "{}\nFor more info on error messages, check: "
#             "https://cloud.google.com/apis/design/errors".format(response.error.message)
#         )

#     # Use OpenCV to decode the image from binary content
#     image_data = io.BytesIO(content)
#     image_array = cv2.imdecode(np.frombuffer(image_data.read(), np.uint8), 1)
    
#     # Display the image using IPython.display
#     display(Image(data=cv2.imencode('.jpg', image_array)[1].tobytes()))
    
#     return description

# for path in image_paths:
#     labels = GetDisplayLabels(path)


## to get more information about what to do with the building materials labeled

def rate_materials_with_gpt(keywords):
    # Create a structured prompt
    prompt = (
        "You are an expert in materials science and you're going to help classify materials based on sustainability. "
        "For each material provided, you will give a rating of its ability to be downcycled, upcycled, recycled, "
        "repurposed, or to go to landfill. Please provide the answers in a structured format as follows:\n"
        "Material: Rating\n\n"
    )
    
    # Add each material to the prompt
    for material in keywords:
        prompt += f"{material}: \n"

    conversation_history = [
        {"role": "system", "content": "You are an assistant knowledgeable in material sustainability."},
        {"role": "user", "content": prompt}
    ]

    response = openai.ChatCompletion.create(
        model="gpt-4",  # Update to the model you are using
        messages=conversation_history,
        max_tokens=50 * len(keywords)  # Adjust tokens based on the number of keywords
    )

    # Extract the assistant's reply from the response
    assistant_reply = response.choices[0].message['content'].strip()
    print(assistant_reply)

    # Parse the assistant's reply
    material_ratings = {}
    for line in assistant_reply.split('\n'):
        parts = line.split(':')
        if len(parts) == 2 and parts[0].strip() in keywords:
            material, rating = parts
            material_ratings[material.strip()] = rating.strip()

    # Convert the parsed data to a Pandas DataFrame
    df = pd.DataFrame(list(material_ratings.items()), columns=['Material', 'Rating'])
    return df


# Example usage
df = rate_materials_with_gpt(building_material_keywords)
