import openai
import cv2
import base64
client = openai.OpenAI()

# This line initializes video capture using OpenCV. It opens the video file bison.mp4. The VideoCapture object, named video, is used to capture frames from this video file.
video = cv2.VideoCapture("bison.mp4")

# Here, an empty list named base64Frames is created. This list will be used to store the frames of the video after they are converted to base64 format
base64Frames = []
# Iterate through all frames of a video in OpenCV.
while video.isOpened():
    #  is called to read the next frame from the video. It returns two values: success (a boolean indicating if a frame was successfully read) and frame (the actual frame read from the video)
    success, frame = video.read()
    if not success:
        break
    # encodes the frame into JPEG format. The cv2.imencode function returns a tuple where the first element is ignored (using _ as a placeholder) and the second element is buffer, which contains the JPEG encoded byte data of the frame
    _, buffer = cv2.imencode(".jpg", frame)
    # first uses base64.b64encode to encode the buffer (JPEG encoded frame) into base64 format. Then, it decodes the base64 encoded data to a UTF-8 string using 
    # his base64 encoded string is appended to the base64Frames list.
    base64Frames.append(base64.b64encode(buffer).decode("utf-8"))
video.release()

response = client.chat.completions.create(
    model="gpt-4-vision-preview",
    # pass the first 5 frames
    # [{"image": frame} for frame in base64Frames[0:5]]
    # This is a list comprehension, a concise way to create lists in Python. It is iterating over the first five elements of the base64Frames list (as indicated by base64Frames[0:5] which slices the first five elements).
    # For each element frame in the slice of base64Frames, it creates a new dictionary with a single key-value pair: "image": frame. The frame variable here represents the base64 encoded string of an image frame (as processed in your previous code snippet).
    # Essentially, this list comprehension is transforming each of the first five base64 encoded frames into dictionaries with the format {"image": frame} and adding them to the content list.
    messages=[{"role": "user", "content": [{"image": frame} for frame in base64Frames[0:5]]}]
)

print(response.choices[0].message.content)