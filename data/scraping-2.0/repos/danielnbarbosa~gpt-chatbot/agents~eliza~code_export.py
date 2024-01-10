import os
import openai

openai.api_key = os.getenv("OPENAI_API_KEY")

start_sequence = "\nEliza:"
restart_sequence = "\n\nPerson:"

response = openai.Completion.create(
  model="text-davinci-002",
  prompt="How do you do?  Please tell me your problem.\n\nPerson: I am unhappy.\nEliza: How do you feel about being unhappy?\n\nPerson: I need some help, that much seems certain.\nEliza: Would it really help you to get some help?\n\nPerson: Perhaps I could learn to get along with my mother.\nEliza: Good family relations are important.\n\nPerson: My mother takes care of me.\nEliza: Why do you think your mother takes care of you?",
  temperature=0.7,
  max_tokens=256,
  top_p=1,
  frequency_penalty=0,
  presence_penalty=0
)
