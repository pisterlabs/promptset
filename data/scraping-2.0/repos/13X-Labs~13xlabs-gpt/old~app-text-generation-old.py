from flask import Flask, request, jsonify
from decouple import config
import openai
import tensorflow as tf

# Declare Flask application
app = Flask(__name__)

# Set OpenAI API key
openai.api_key = config('OPENAI_API_KEY')

# Path to GPT-2 model
model_path = "data/small-117M"

# Load GPT-2 model
model = tf.saved_model.load(model_path)

# Define input and output tensors
input_text = model.signatures["serving_default"].inputs[0]
output = model.signatures["serving_default"].outputs["output_0"]

# Define API endpoint
@app.route('/api/complete', methods=['POST'])
def complete():
    data = request.get_json()
    prompt = data['prompt']
    length = data.get('length', 100)
    temperature = data.get('temperature', 0.5)

	# Use GPT-2 model to generate text
    output_text = model(input_text=tf.constant([prompt]), length=tf.constant(length), temperature=tf.constant(temperature))["output_0"].numpy()[0].decode('utf-8')

	# Return generated text as JSON
    return jsonify({
        "output": output_text
    })

# Start the application
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
