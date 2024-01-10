from flask import Flask, request, jsonify
from dotenv import load_dotenv
from prompt.gpt_4v_prompt import gpt_4v_action_prompt
import os
import base64
import requests
from datetime import datetime
from openai import OpenAI
import re
import json

# Load environment variables from .env file
load_dotenv(override=True)

app = Flask(__name__)

@app.route('/process_query', methods=['POST'])
def process_query():
    request_data = request.get_json()
    print("Request data:", request_data)
    query_string = request_data.get('query_string')
    img_url = request_data.get('img_url')
    element_centers = request_data.get('element_centers')
    current_link = request_data.get('current_link')
    log = request_data.get('log')

    if not all([query_string, img_url, element_centers, current_link]) or log is None:
        print("Error: Missing required data in request.")
        return jsonify({"error": "Missing data"}), 400

    api_key = os.getenv('OPENAI_KEY')

    if not api_key:
        print("Error: OPENAI_KEY is not set in environment variables.")
        return jsonify({"error": "Missing OPENAI_KEY"}), 400

    client = OpenAI(api_key=api_key)

    element_prompt = ""
    for element_id, element_info in element_centers.items():
        element_prompt += f"Element {element_id} is a {element_info['tag']}"
        if 'link' in element_info and element_info['link']:
            element_prompt += f" with link {element_info['link']}\n"
        else:
            element_prompt += "\n"

    prompt = gpt_4v_action_prompt.format(query=query_string, element_info=element_prompt, current_link=current_link, log=log)
    print("Generated prompt:", prompt)
    # Parse down the name after /main/ in the URL
    image_name = img_url.split('/main/')[-1]
    print("Image name extracted:", image_name)
    img_url = "https://raw.githubusercontent.com/Ruijian-Zha/My_Image/main/" + image_name
    response = client.chat.completions.create(
        model="gpt-4-vision-preview",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": img_url,
                        },
                    },
                ],
            }
        ],
        max_tokens=300,
    )

    message = response.choices[0].message.content
    input_str = message

    match = re.search(r'```json\n(.+?)\n```', input_str, re.DOTALL)

    if match:
        json_str = match.group(1)
        try:
            parsed_json = json.loads(json_str)
        except json.JSONDecodeError as e:
            print(f"Error parsing JSON: {e}")
            return jsonify({"error": "Error parsing JSON"}), 500
    else:
        print("No JSON found in the string")
        return jsonify({"error": "No JSON found in the string"}), 500

    return jsonify(parsed_json)


@app.route('/upload', methods=['POST'])
def upload_to_github():
    """
    Handles a POST request to the '/upload' route, retrieves data from the request, checks for necessary data, and calls the function to upload a base64 encoded image to GitHub.

    The function has no explicit parameters. It retrieves 'image' (base64 encoded), 'username', 'repo', and 'token' from the request data and environment variables.

    Returns:
        A JSON response with either the URL of the uploaded image upon a successful upload or an error message indicating failure.

    Exceptions:
        Catches any exceptions during the image upload and returns a JSON response with the error message.
    """

# Retrieve the file from the request
    if 'image' not in request.files:
        return jsonify({"error": "No image file provided"}), 400
    file = request.files['image']
    encoded_image = base64.b64encode(file.read()).decode('utf-8')
    username = os.getenv('USERNAME')
    repo = os.getenv('REPO')
    token = os.getenv('TOKEN')

    if not encoded_image or not username or not repo or not token:
        print(request.form)
        return jsonify({"error": "Missing data"}), 400

    # Call the existing function to upload the image
    try:
        # Generate a unique filename based on the current timestamp
        image_filename = datetime.now().strftime('%Y%m%d%H%M%S') + ".png"
        upload_url = upload_image_to_github_base64(encoded_image, username, repo, token, image_filename)
        if upload_url.startswith("Error"):
            return jsonify({"error": upload_url}), 500
        print("Upload URL:", upload_url)
        return jsonify({"img_url": upload_url})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

def upload_image_to_github_base64(encoded_image, username, repo, token, image_filename):
    """
    Upload a base64 encoded image to a specified GitHub repository.

    Args:
    encoded_image (str): The base64 encoded image string.
    username (str): Your GitHub username.
    repo (str): The name of the GitHub repository.
    token (str): Your GitHub Personal Access Token.
    image_filename (str): The filename to use for the uploaded image.

    Returns:
    str: The URL of the uploaded file if successful, else an error message.
    """
    url = f"https://api.github.com/repos/{username}/{repo}/contents/{image_filename}"
    headers = {"Authorization": f"token {token}"}
    payload = {"message": "Upload image", "content": encoded_image}

    response = requests.put(url, json=payload, headers=headers)

    if response.status_code == 201:
        # Construct the user-facing GitHub URL for the image
        github_url = f"https://github.com/{username}/{repo}/blob/main/{image_filename}"
        return github_url
    else:
        return "Error: " + response.json().get('message', 'Unable to upload file')

if __name__ == '__main__':
    app.run(debug=True)