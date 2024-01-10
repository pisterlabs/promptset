# Define a function to process the image and get the text
import base64

import cv2
import openai

from CompleteModels.reuse import capture_image, speak

openai.api_key = "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"
def image_to_text(image_path):
    # Load the image using OpenCV
    img = cv2.imread(image_path)
    speak("processing....")

    # Preprocess the image if necessary
    # Resize the image to a smaller size
    speak("resizing  ...")
    img = cv2.resize(img, (128,128))
    speak("convertiion complete...")
    # Convert the image to base64-encoded string
    img_base64 = base64.b64encode(cv2.imencode('.jpeg', img)[1]).decode()
    speak("convertiion complete...")

    speak("sending data to openai...")
    # Convert the image to text using the OpenAI API
    response = openai.Image.create(
        prompt=img_base64,
        n=1,
        size="1024x1024",
        response_format="text"
    )
    text = response["data"][0]["text"]
    speak("processing successfull")
    # Return the text
    return text

path = capture_image()

imagetext = image_to_text(path)
speak(imagetext)

# # Call the function with an example image path
# result = image_to_text("example_image.jpg")
# print(result)