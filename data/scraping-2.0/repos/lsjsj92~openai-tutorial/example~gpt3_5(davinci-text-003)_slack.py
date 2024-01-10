
import os
import openai
import argparse
from dotenv import load_dotenv
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError


class OpenAIGpt:
  def run(self, msg):
    openai.api_key = os.getenv("OPENAI_API_KEY")
    response = openai.Completion.create(
      model="text-davinci-003",
      prompt=f"{msg}.",
      temperature=0.3,
      max_tokens=512,
      top_p=1,
      frequency_penalty=0.0,
      presence_penalty=0.0,
      stop=None
    )
    
    return response.choices[0].text.strip()


load_dotenv()

client = WebClient(token=f"{os.getenv('slack_bot')}")
channel_id = os.getenv('slack_ch')

try:
    result = client.conversations_history(channel=channel_id)

    conversation_history = result["messages"]
    openai_gpt = OpenAIGpt()
    ans = openai_gpt.run(conversation_history[0]['text'])
    print("conversation_history[0]['text'] : ", conversation_history[0]['text'])
    print(ans)
    client.chat_postMessage(channel='#alert_msg', text=f"{ans}")
except SlackApiError as e:
    print("Error creating conversation: {}".format(e))

