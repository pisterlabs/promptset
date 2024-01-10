# WRITE YOUR CODE HERE
import openai
import os
# Replace 'your_api_key' with your actual API key
openai.api_key = os.getenv('OPENAI_KEY')

def generate_website(prompt):
  response=openai.ChatCompletion.create(model="gpt-3.5-turbo",
  messages=[{"role": "system", "content": "You are a html expert you are going to write a html file"},
        {"role": "user", "content": prompt}])
  website_code = response['choices'][0]['message']['content'].strip()
  return website_code

#generate_website("You are a HTML expert you are going to write a HTML file. In this HTML file, we define a basic structure for our website that includes a title, a header, a form for user input, and a script tag that loads our JavaScript code. We also include a link to a CSS file (style.css) that defines the styling for our website.")