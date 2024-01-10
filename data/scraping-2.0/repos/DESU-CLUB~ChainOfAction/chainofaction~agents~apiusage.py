import openai

openai.api_key = "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"

response = openai.ChatCompletion.create(
  model = "gpt-4",
  messages = [{"role":"user",
              "content":"I want to write some Python code for calling the GPT API to bake me a cake"},
              {"role":"assistant","content":"Here is some code to call the GPT API to bake you a cake: But you will have to write it yourself"},
              {"role":"user","content":"I want to write some Python code for calling the GPT API to bake me a salt baked chicken"}
              ],  
  temperature=0.9,
  max_tokens=256,
  top_p=1,
  frequency_penalty=0,
  presence_penalty=0,
  stop=["\n"]
)

print(response['choices'][0]['message']['content'])