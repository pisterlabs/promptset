import os
import openai
import sys

# get task title from first input argument
#if empty set default task title
task_title = sys.argv[1] if len(sys.argv) > 1 else 'Elaborar una tarea'

openai.api_key = "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" #os.getenv("OPENAI_API_KEY")



response = openai.Completion.create(
  engine="text-davinci-002",
  prompt="Asistente: Hola, ¿que tarea deseas agregar al listado?\nYo: " + task_title + "\nAsistente:",
  temperature=0.1,
  max_tokens=256,
  top_p=1,
  frequency_penalty=0,
  presence_penalty=0
)

# print response
print(response)
print(response.choices[0].text)
