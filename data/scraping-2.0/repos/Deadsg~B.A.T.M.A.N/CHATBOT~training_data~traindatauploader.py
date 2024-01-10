from openai import OpenAI
client = OpenAI()

client.files.create(
  file=open("Batman_AI.jsonl", "rb"),
  purpose="fine-tune"
)
