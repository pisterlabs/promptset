from flask import Flask, request, jsonify
import openai

app = Flask(__name__)

def init_app(app):
  @app.route('/process_text', methods=['POST'])
  def process_text():
      text_content = request.json['textAreaContent']
  
      prompt = r"""
      You are an assistant that takes the user message and fits the message into the following JSON structure.
      You may fill the skillset using the other information provided by the user.
      {
        Personal: {
          Name: "",
          Email: "",
          Number: "",
          Linkedin: "",
          Github: "",
          Portfolio: "",
        },
        Experience: [
         {
            Name: "",
            Title: "",
            Location: "",
            Description: "",
            StartDate: "",
            EndDate: "",
          }
        ],
        Education: [{
          Name: "",
          Location: "",
          Degree: "",
          Field: "",
          Score: 0,
          StartDate: "",
          EndDate: "",
        }],
        Projects: [{
          Name: "",
          Technologies: "",
          Link: "",
          Description: "",
        }],
        Skillset: {
          languages: "",
          libraries: "",
          tools: "",
        },
        Certifications: [{
          Name: "",
          Link: "",
          Issuer: "",
        }],
      }
      """ 
      user = text_content
  
      client = openai.Client(api_key="XXXXX")
  
      completion = client.chat.completions.create(
          model="gpt-3.5-turbo-1106",
          response_format={"type": "json_object"},
          messages=[
              {"role": "system", "content": prompt},
              {"role": "user", "content": user}
          ],
          temperature=0.4
      )
  
      return completion.choices[0].message.content