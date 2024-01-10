import os
import openai

openai.api_key = os.getenv("OPENAI_API_KEY")

start_sequence = "\nA:"
restart_sequence = "\nQ:"

response = openai.Completion.create(
  engine="davinci",
  prompt="Q: Ask Constance if we need some bread\nA: send-msg `find constance` Do we need some bread?\nQ: Send a message to Greg to figure out if things are ready for Wednesday.\nA: send-msg `find greg` Is everything ready for Wednesday?\nQ: Ask Ilya if we're still having our meeting this evening\nA: send-msg `find ilya` Are we still having a meeting this evening?\nQ: Contact the ski store and figure out if I can get my skis fixed before I leave on Thursday\nA: send-msg `find ski store` Would it be possible to get my skis fixed before I leave on Thursday?\nQ: Thank Nicolas for lunch\nA: send-msg `find nicolas` Thank you for lunch!\nQ: Tell Constance that I won't be home before 19:30 tonight — unmovable meeting.\nA: send-msg `find constance` I won't be home before 19:30 tonight. I have a meeting I can't move.\nQ: ",
  temperature=0.5,
  max_tokens=100,
  top_p=1,
  frequency_penalty=0.2,
  presence_penalty=0,
  stop=["\n"]
)
