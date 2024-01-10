from openai import OpenAI
# import os
# from dotenv import load_dotenv

def lambda_handler(event, context):
  # load_dotenv('./.env')
  # client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
  client = OpenAI()
  context = event['context']
  numQuestions = ("{} ".format(event['numQuestions'])) if 'numQuestions' in event else "1 "
  
  response = client.chat.completions.create(
    model="gpt-3.5-turbo-1106",
    response_format={ "type": "json_object" },
    messages=[
      {"role": "system", "content": "Form " + numQuestions + "formal question(s) based solely on the information in user content. Add multiple choice answers. Return in JSON with one key for a JSON list called questions with each list element having the keys: question, a, b, c, d, and correctAnswerChoice"},
      {"role": "user", "content": context}
    ]
  )
  
  return {
    'statusCode': 200,
    'body': response.choices[0].message.content
  }