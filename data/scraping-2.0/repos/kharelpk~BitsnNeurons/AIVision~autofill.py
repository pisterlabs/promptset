### Take a screenshot image and identify buttons that can be clicked

# first load the openai client
import os
from dotenv import load_dotenv
load_dotenv()
from openai import OpenAI
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
import time

import pyautogui
im1 = pyautogui.screenshot()
im1.save('screenshot.png')
time.sleep(1)

# Load image and make an openai api call
import base64

# Function to encode the image
def encode_image(image_path):
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')
  
# Image path
image_path = "screenshot.png"
base64_image = encode_image(image_path)

response = client.chat.completions.create(
  model="gpt-4-vision-preview",
  messages=[
    {
      "role": "user",
      "content": [
        {"type": "text", "text": "Given the dialog box, from top to bottom list the boxes that need to be filled. Only return JSON with a list of input_fields. "},
        {
          "type": "image_url",
          "image_url": {
            "url": f"data:image/png;base64,{base64_image}",
          },
        },
      ],
    }
  ],
  max_tokens=500,
)
print(response.choices[0].message.content)


import json
json_text = response.choices[0].message.content
json_data = json.loads(json_text[7:-3])


### Identify the locations of these buttons using traditional OCR mapping techniques

import pytesseract
from PIL import Image, ImageDraw
image_pil_2 = Image.open("my_screenshot_1.png").convert("RGB")


def find_text_and_draw_boxes(image, search_texts):
    # Load the image
    img = image
    draw = ImageDraw.Draw(img)

    # Use pytesseract to do OCR on the image
    text_data = pytesseract.image_to_data(img)

    # Dictionary to store coordinates for each text
    coordinates_dict = {text: [] for text in search_texts}

    # Iterate through the data
    for line in text_data.split('\n'):
        parts = line.split('\t')  # Split by tab character
        if len(parts) == 12:
            ocr_text = parts[11]
            for search_text in search_texts:
                if search_text in ocr_text:
                    try:
                        x, y, w, h = map(int, parts[6:10])
                        coordinates_dict[search_text].append((x, y, w, h))
                        # Draw a rectangle around the found text
                        # draw.rectangle([x, y, x + w, y + h], outline="green", width=2)
                    except ValueError:
                        print("Error parsing line:", line)

    # Show the image
    # img.show()

    return coordinates_dict


# Example usage
search_texts = json_data['input_fields'] 
coordinates = find_text_and_draw_boxes(image_pil_2, search_texts)

# Print coordinates for each text
for text, coords in coordinates.items():
    print(f'Coordinates for "{text}": {coords}')


def update_json_with_coordinates(json_data, coordinates):
    for field in json_data:
        for idx, item in enumerate(json_data[field]):
            # Check if the item is in coordinates and if the coordinates list is not empty
            if item in coordinates :
                # Update the item with its text and coordinates
                json_data[field][idx] = {"text": item, "coordinates": coordinates[item]}
    return json_data


# Update the JSON data with coordinates
updated_json_data = update_json_with_coordinates(json_data, coordinates)

# Print the updated JSON data
print(json.dumps(updated_json_data, indent=4))



### Go to screeen coordinates and type
input_data={'Email':'hello@hello.com',
            'Password':'xxxxxxx'}
import pyautogui

# Move cursor, point, click, fill the documents
import time
for val in json_data['input_fields']:
    print(val['coordinates'][0])
    x, y, w, h = val['coordinates'][0]
    pyautogui.moveTo(int(x/2)+w,int(y/2)+h)
    pyautogui.click()
    pyautogui.click()
    pyautogui.typewrite(input_data[val['text']])
    time.sleep(0.5)
