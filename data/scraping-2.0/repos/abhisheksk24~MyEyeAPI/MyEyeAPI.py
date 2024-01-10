from flask import Flask, request, jsonify
import cv2
import base64
import tempfile
import os
from openai import OpenAI
from flask_cors import CORS, cross_origin
 
app = Flask(__name__)

# Get the OpenAI API key from the environment variable
openai_api_key = os.environ.get('newkey')

 
# Initialize the OpenAI client globally
openai_client = OpenAI(api_key=openai_api_key)
assistant_id_1= ""
thread_id_1= ""

file = openai_client.files.create(
file=open("products.txt", "rb"),
purpose='assistants'
)

assistant = openai_client.beta.assistants.create(
    instructions="You are a helpful chatbot. Treat file knowlege as a database of the store so that whenever asked about products you can use this a knowledge. While responding to queries dont mention about uploaded file",
    model="gpt-4-1106-preview",
    tools=[{"type": "retrieval"}],
    file_ids=[file.id]
)
assistant_id_1 = assistant.id
thread = openai_client.beta.threads.create()
thread_id_1 = thread.id

# Video processing endpoint
@app.route('/process-video', methods=['POST'])
@cross_origin(origin='*')
def process_video():
    # Check if 'video' key is in the files
    if 'video' not in request.files:
        return jsonify({"error": "No video file provided"}), 400
 
    video_file = request.files['video']
 
    # Save the video to a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp_video:
        video_file.save(temp_video.name)
        temp_video.close()
 
        # Open the video file
        video = cv2.VideoCapture(temp_video.name)
 
        # Check if video is opened successfully
        if not video.isOpened():
            return jsonify({"error": "Failed to open video file"}), 400
 
        # Process the first frame of the video
        success, frame = video.read()
        if not success:
            return jsonify({"error": "Failed to read video frame"}), 400
 
        # Convert the frame to base64 for processing
        _, buffer = cv2.imencode('.jpg', frame)
        base64_frame = base64.b64encode(buffer).decode('utf-8')
 
        # Create the payload for OpenAI
        prompt_messages = [
            {
                "role": "user",
                "content": [
                     "These are frames from a video that I want to upload. Generate a compelling description for a visually challenged person to understand. Make it short and concise. Act as an interpreter who is describing the scene for a blind person. Detect the depth in the image and provide user with distance between object and camera lence . Dont start the responce by saying ' The uploaded image states that' . Dont mention Visually challanged person in responce",
                    {"image": base64_frame, "resize": 768},
                ],
            },
        ]
        params = {
            "model": "gpt-4-vision-preview",
            "messages": prompt_messages,
            "max_tokens": 200,
        }
 
        # Call the OpenAI API
        result = openai_client.chat.completions.create(**params)
        response_text = result.choices[0].message.content
 
        # Return the response
        return jsonify({"description": response_text})
 
# Image processing endpoint
@app.route('/process-image', methods=['POST'])
@cross_origin(origin='*')
def process_image():
    # Check if 'video' key is in the files
    if 'video' not in request.files:
        return jsonify({"error": "No video file provided"}), 400
 
    video_file = request.files['video']
 
    # Save the video to a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp_video:
        video_file.save(temp_video.name)
        temp_video.close()
 
        # Open the video file
        video = cv2.VideoCapture(temp_video.name)
 
        # Check if video is opened successfully
        if not video.isOpened():
            return jsonify({"error": "Failed to open video file"}), 400
 
        # Process the first frame of the video
        success, frame = video.read()
        if not success:
            return jsonify({"error": "Failed to read video frame"}), 400
 
        # Convert the frame to base64 for processing
        _, buffer = cv2.imencode('.jpg', frame)
        base64_frame = base64.b64encode(buffer).decode('utf-8')
 
        # Create the payload for OpenAI
        prompt_messages = [
            {
                "role": "user",
                "content": [
                     "This is the photo I want to upload. If the newspaper is detected inside the image  summrize the important news and provide summary of the said news while keeping in mind all of this should happen in the orignal language of the newspaper. If the detected image is of a book identify the text in the book and list it out.",
                    {"image": base64_frame, "resize": 768},
                ],
            },
        ]
        params = {
            "model": "gpt-4-vision-preview",
            "messages": prompt_messages,
            "max_tokens": 200,
        }
 
        # Call the OpenAI API
        result = openai_client.chat.completions.create(**params)
        response_text = result.choices[0].message.content
 
        # Return the response
        return jsonify({"description": response_text})


# Chat endpoint
@app.route('/chat', methods=['POST'])
@cross_origin(origin='*')
def chat():
   # Initial system message
    data = request.json
    msg = data.get("message")
    print(data.get("message"))

    message = openai_client.beta.threads.messages.create(
        thread_id=thread_id_1,
        role="user",
        content=msg
    )
    run = openai_client.beta.threads.runs.create(
        thread_id=thread_id_1,
        assistant_id= assistant_id_1,
        instructions="You are a helpful chatbot."
    )

    # Retrieve response
    run = openai_client.beta.threads.runs.retrieve(
        thread_id=thread_id_1,
        run_id=run.id
    )

    # Wait for completion
    while(run.status != 'completed'):
        run = openai_client.beta.threads.runs.retrieve(
            thread_id=thread_id_1,
            run_id=run.id
        )

    messages = openai_client.beta.threads.messages.list(
        thread_id=thread_id_1
    )
    response = ""
    for msg in messages:
        response = msg.content[0].text.value
        if(True):
            break
    # # Return response to user
    return jsonify({"response": response})

if __name__ == '__main__':
    app.run(debug=True)
