import openai

openai.api_key = "your-api-key"

def process_user_input(user_input):
 response = openai.Completion.create(
 engine="text-davinci-002",
 prompt=user_input,
 temperature=0.5,
 max_tokens=1024,
 top_p=1,
 frequency_penalty=0,
 presence_penalty=0
 )

 response_text = response["choices"][0]["text"]

 if response_text.startswith("I'm sorry"):
 return "I'm sorry, I didn't understand your question. Could you please rephrase it?"
 else:
 return response_text

