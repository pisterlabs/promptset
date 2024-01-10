#from openai import OpenAI

class MyClient:
    def __init__(self):
      _client = OpenAIClient(api_key=os.environ['OPENAI_API_KEY'])

    def request_gpt(self, data):
      _response = _client.chat.completions.create(
        model=data["model"],
        messages=data["messages"],
      )
      return _response.choices[0].message.content
    




# Function to format the dictionary based on inputs
def format_dict(deliverable, subject, content, model, format="HTML"):
    userPrompt = f"Create a {deliverable} for high school teachers specializing in {subject}, focusing on the topic: '{content}'."
    userPrompt += " This deliverable should serve as a guide for designing engaging classroom experiences."
    outputFormat = f"Please format the response in {format}."
    data = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You are the most advanced AI-based NLP engine acting as an expert on the user's prompt topic and as a prompt enhancer"},
            {"role": "system", "content": "You must take user's prompt and provide an improved prompt at the end of your answer."},
            {"role": "user", "content": userPrompt},
            {"role": "user", "content": outputFormat}
        ]
    }
    return data