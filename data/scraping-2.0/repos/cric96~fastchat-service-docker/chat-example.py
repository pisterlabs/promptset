import openai
openai.api_key = "EMPTY" # Not support yet
openai.api_base = "http://clusters.almaai.unibo.it:23231/v1"

model = "vicuna"

print("Chat example, write senteces and exit with 'exit'")
current_sentence = ""
while(True):
  current_sentence = input()
  if(current_sentence == "exit"):
    break
  completion = openai.ChatCompletion.create(
    model=model,
    messages=[{"role": "user", "content": current_sentence}]
  )
  # print the completion
  print(completion.choices[0].message.content)
