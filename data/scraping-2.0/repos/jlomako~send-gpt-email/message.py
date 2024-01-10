import smtplib
import os
import openai

# generate message
openai.api_key = os.getenv('OPENAI_API_KEY')
response = openai.Completion.create(
  model="text-davinci-003",
  prompt="write a short text message to someone you love very much",
  temperature=0.7,
  max_tokens=50,
  top_p=1,
  frequency_penalty=0,
  presence_penalty=0
)

message = response["choices"][0]["text"]

# send message to myself
my_email = os.environ['MY_EMAIL']
my_email2 = os.environ['MY_EMAIL2']
pw = os.environ['PASSWORD']

with smtplib.SMTP(host='smtp.gmail.com', port=587) as connection:
    connection.starttls()
    connection.login(user=my_email, password=pw)
    connection.sendmail(from_addr=my_email,
                        to_addrs=my_email2,
                        msg=f'Subject: hello\n\n{message}'
                        )
