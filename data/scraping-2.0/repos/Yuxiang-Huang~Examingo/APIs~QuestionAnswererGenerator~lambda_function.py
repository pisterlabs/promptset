from openai import OpenAI
# import os
# from dotenv import load_dotenv

def lambda_handler(event, context):
  # load_dotenv('./.env')
  # client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
  client = OpenAI()
  context = event['context']
  question = event['question']
  
  response = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[
      {"role": "system", "content": "Answer the question posed by the user based on the following context: " + context},
      {"role": "user", "content": question}
    ]
  )
  
  return {
    'statusCode': 200,
    'body': response.choices[0].message.content
  }