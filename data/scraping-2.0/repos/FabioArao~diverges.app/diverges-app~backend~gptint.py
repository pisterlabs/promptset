#import openai
#openai.my_api_key = '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'

#import openai
from openai import OpenAI
client = OpenAI(api_key="")

response = client.chat.completions.create(
  model="gpt-3.5-turbo-1106",
  response_format={ "type": "json_object" },
  messages=[
    {"role": "system", "content": "who and what he or she does? the following text output JSON."},
    {"role": "user", "content": "Fabio Arão Cloud and infraestructures engineer Highly skilled IT Professional with over 15 years of comprehensive experience in Cloud Computing, Infrastructure Management, and Identity and Access Management (IAM). Proven record in leading and implementing diverse high-stakes IT projects. AWS and Azure certified with strong acumen in managing service delivery, support, and project management in large-scale environments. Adept at building and maintaining robust relationships with stakeholders and teams."}
  ]
)
print(response.choices[0].message.content)