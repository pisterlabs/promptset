import os
import openai
import pyperclip
from notifypy import Notify

# Requires a paid API key: https://platform.openai.com/account/api-keys
openai.api_key = os.getenv("OPEN_API_KEY")

ClipBoard = pyperclip.paste()

StrCount = len(ClipBoard)
CostPerToken = (0.002/1000)
QueryCost = round(((StrCount/(3/4))*CostPerToken), 5)
PrintCost = f"Cost: ${QueryCost:.7f}"
print(PrintCost)

response = openai.Completion.create(
    model="text-davinci-003",
    prompt=f'The following is a conversation with an AI assistant. The assistant is smart, helpful, creative, clever, and very friendly.\n\nHuman: {ClipBoard}\nAI:',
    temperature=0.9,
    max_tokens=150,
    top_p=1,
    frequency_penalty=0.0,
    presence_penalty=0.6,
    stop=[" Human:", " AI:"]
)
ResponseText = response.choices[0].text
# debug
# print(response)
print(ResponseText)
pyperclip.copy(ResponseText)

notification = Notify()
notification.title = "ChatGPT"
notification.message = ResponseText
notification.send()
