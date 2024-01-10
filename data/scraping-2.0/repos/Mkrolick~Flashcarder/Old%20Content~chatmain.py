import os
import openai
import dotenv
import prompts

dotenv.load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")

start_sequence = "\nAI:"
restart_sequence = "\nHuman: "

response = openai.ChatCompletion.create(
  model="gpt-4",
  messages= [{"role": "system", "content": "You produce flashcard from a specific book. You produce highly detailed flash cards with a term name and a definition. You produce around 20 flashcards per book chapter."}, {"role": "user", "content": prompts.book_summary_prompt}],
)




# save out put to a text file
with open('output.txt', 'w') as f:
    f.write(response["choices"][0]["message"]["content"])
