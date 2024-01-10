import openai, os, json

def generate_openai_response(text):
  openai.api_key = os.environ.get("OPENAI_API_KEY")
  response = openai.ChatCompletion.create(
  model="gpt-3.5-turbo-16k",
  messages=[
        {"role": "system", "content": "Você é um assistente de advocacia que vai me ajudar a entender documentos juridicos."},
        {"role": "user", "content": text},
    ])
    
  json_string = response.choices[0].message.content.strip('"')
  json_obj = json.loads(json_string)
  return json_obj