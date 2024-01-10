from flask import Flask, request, jsonify
import openai
import os

app = Flask(__name__)

# Load OpenAI API key from environment variables for security
openai.api_key = os.getenv('OPENAI_API_KEY')

@app.route('/generate-image', methods=['POST'])
def generate_image():
    data = request.json
    selected_item = data['selectedItem']
    text = data['text']

    try:
        # Generating an image using DALL-E
        response = openai.Image.create(
            prompt=f"{selected_item} {text}",
            n=1,  # Number of images to generate
            size="1024x1024"  # Size of the image
        )

        # Assuming the response contains a URL to the generated image
        image_url = response['data'][0]['url']

        return jsonify({'image_url': image_url})

    except Exception as e:
        print(f"An error occurred: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
