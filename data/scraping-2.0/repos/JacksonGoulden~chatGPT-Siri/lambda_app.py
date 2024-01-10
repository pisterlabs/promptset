# This lambda function is used to receive the phrase given by the SIRI user, and then send it to chatGPT to get a response.

import openai
import json

def lambda_handler(event, context):
  
  # Log the event details - for viewing later in CloudWatch
  print(json.dumps(event))
  
  # Log the phrase given by the SIRI user - for viewing later in CloudWatch
  print(event["headers"]["gpt-phrase"])
  
  # Sets openai credentials. The application will fail without changing these values.
  # These are found at:
  # https://platform.openai.com/account/org-settings
  # https://platform.openai.com/account/api-keys
  openai.organization = "INSERT ORGINIZATION NAME HERE" # eg "org-jhvfdshfvadshvsdauhvasdhvhvsajf"
  openai.api_key = "INSERT KEY FROM OPENAI HERE" # eg "sk-shdygfADSAWDASDAWDuyhfgdsfgsfdh"
  
  # Checks to see if the credentials have been changed. If not, the application will fail.
  if openai.organization == "INSERT ORGINIZATION NAME HERE":
    return {
        'statusCode': 400,
        'body': "OpenAI credentials are not set. Run 'terraform destroy' then modify the credentials in lambda_app.py before re-deploying"
    }

  if openai.api_key == "INSERT KEY FROM OPENAI HERE":
    return {
        'statusCode': 400,
        'body': "OpenAI credentials are not set. Run 'terraform destroy' then modify the credentials in lambda_app.py before re-deploying"
    }
  
  # Retrieves the phrase given to siri through the "gpt-phrase" header and saves it as a variable
  text_to_ask = event["headers"]["gpt-phrase"]
  
  # Creates a request to chatGPT 
  completion = openai.ChatCompletion.create(
  model="gpt-3.5-turbo", # Sets the AI model to use. See here for a list https://platform.openai.com/docs/models
  messages=[
    {"role": "user", "content": text_to_ask} # The body of the message, with the content to send as the variable saved from SIRI
    ]
  )
  
  # Save the chatGPT response as a variable. This will be returned
  notification = str(completion.choices[0].message.content)
  
  # Return a 200 success code and the chatGPT response as the body
  return {
    'statusCode': 200,
    'body': notification
   }
