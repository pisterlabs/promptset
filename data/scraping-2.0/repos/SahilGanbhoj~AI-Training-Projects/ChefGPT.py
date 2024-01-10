# This code imports the necessary modules and libraries for a Flask web application that will render templates using OpenAI and accept user input through HTML forms.

import openai

import os

from flask import Flask, render_template_string, request

# Sets the API key for the OpenAI API using the value of the 'OPENAI_API_KEY' environment variable.

openai.api_key = os.environ['OPENAI_API_KEY']

# This code uses the OpenAI API to generate a tutorial for making a recipe using a list of components provided as an argument.3

def generate_tutorial(components):

 response = openai.ChatCompletion.create(

  model="gpt-3.5-turbo",

  messages=[{

   "role": "system",

   "content": "You are a helpful assistant"

  }, {

   "role":

   "user",

   "content":

   f"Suggest a recipe using the items listed as available. Make sure you have a nice name for this recipe listed at the start. Also, include a funny version of the name of the recipe on the following line. Then share the recipe in a step-by-step manner. In the end, write a fun fact about the recipe or any of the items used in the recipe. Here are the items available: {components}, Haldi, Chilly Powder, Tomato Ketchup, Water, Garam Masala, Oil"

  }])

 return response['choices'][0]['message']['content']

# Create a Flask web application object named app and define a route for the root URL that responds to GET and POST requests

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])

# This code defines a function that generates a tutorial based on user input obtained through a POST request.

def hello():

 output = ""

 if request.method == 'POST':

  components = request.form['components']

  output = generate_tutorial(components)

# This is a HTML template for a Custom Recipe Generator web page. It includes a form for users to input a list of ingredients/items they have, and two JavaScript functions for generating a recipe based on the input and copying the output to the clipboard. The template uses the Bootstrap CSS framework for styling.

 return render_template_string('''

 <!DOCTYPE html>

 <html>

 <head>

  <title>Infinite Project Generator</title>

    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0-alpha1/dist/css/bootstrap.min.css" rel="stylesheet">

  <script>

  async function generateTutorial() {

   const components = document.querySelector('#components').value;

   const output = document.querySelector('#output');

   output.textContent = 'Cooking a recipe for you...';

   const response = await fetch('/generate', {

    method: 'POST',

    body: new FormData(document.querySelector('#tutorial-form'))

   });

   const newOutput = await response.text();

   output.textContent = newOutput;

  }

  function copyToClipboard() {

   const output = document.querySelector('#output');

   const textarea = document.createElement('textarea');

   textarea.value = output.textContent;

   document.body.appendChild(textarea);

   textarea.select();

   document.execCommand('copy');

   document.body.removeChild(textarea);

   alert('Copied to clipboard');

  }

  </script>

 </head>

 <body>

  <div class="container">

   <h1 class="my-4">Custom Recipe Tutorial Generator</h1>

   <form id="tutorial-form" onsubmit="event.preventDefault(); generateTutorial();" class="mb-3">

    <div class="mb-3">

     <label for="components" class="form-label">Ingredients / Items:</label>

     <input type="text" class="form-control" id="components" name="components" placeholder="Enter the list of Ingredients or items you have e.g. Bread, jam, potato etc. " required>

    </div>

    <button type="submit" class="btn btn-primary">Share with me a tutorial</button>

   </form>

   <div class="card">

    <div class="card-header d-flex justify-content-between align-items-center">

     Output:

     <button class="btn btn-secondary btn-sm" onclick="copyToClipboard()">Copy</button>

    </div>

    <div class="card-body">

     <pre id="output" class="mb-0" style="white-space: pre-wrap;">{{ output }}</pre>

    </div>

   </div>

  </div>

 </body>

 </html>

 ''',

                output=output)

# This code defines a route for the URL "/generate" that only accepts POST requests.

@app.route('/generate', methods=['POST'])

# This code defines a function 'generate' that takes a POST request containing a 'components' field and returns the result of calling the 'generate_tutorial' function with the provided components as input.

def generate():

 components = request.form['components']

 return generate_tutorial(components)

# This code snippet starts the Flask application if the script is being run directly as the main program, running on the IP address '0.0.0.0' and port number 8080.

if __name__ == '__main__':

 app.run(host='0.0.0.0', port=8080)