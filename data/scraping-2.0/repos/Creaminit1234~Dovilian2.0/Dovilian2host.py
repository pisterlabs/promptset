from flask import Flask, render_template, jsonify, request
import openai

app = Flask(__name__)
f = open("openai_key.Dovilian2CONFIG-V1", "r")
OpenAiKey = f.read()
openai.api_key = OpenAiKey

def generateImageDovilian1926(promptDovilianImageGeneration1926):
	GeneratedImageDovilianImageGeneration1926 = openai.Image.create(prompt=promptDovilianImageGeneration1926, model="image-alpha-001", n=1)
	GeneratedImageDovilian1926 = responseDovilian1926['data'][0]['url']
	return GeneratedImageDovilian1926

@app.route('/')
def index():
    return render_template('dovilianImageGeneration.html')

@app.route('/', methods=['GET', 'POST'])
def generate_image():
    promptDovilianImageGeneration1926 = request.form.get('promptDovilianImageGeneration1926')
    response = openai.Image.create(
    model="image-alpha-001",  
    prompt=promptDovilianImageGeneration1926,
    size="512x512")
    GeneratedImageDovilian1926 = response['data'][0]['url']

    return render_template("GeneratedImage.html", img=GeneratedImageDovilian1926)

if __name__ == '__main__':
    app.run("0.0.0.0", port=1926, debug=True)
