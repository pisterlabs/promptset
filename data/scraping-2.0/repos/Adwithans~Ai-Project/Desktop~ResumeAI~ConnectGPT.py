from openai import OpenAI
import requests

client = OpenAI(api_key = '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')
gpt_assistant_prompt = "You are a " + input ("Who should I be, as I answer your prompt?") 
gpt_user_prompt = input ("What prompt do you want me to do?") 
gpt_prompt = gpt_assistant_prompt, gpt_user_prompt
print(gpt_prompt)

message=[{"role": "assistant", "content": gpt_assistant_prompt}, {"role": "user", "content": gpt_user_prompt}]
temperature=0.2
max_tokens=256
frequency_penalty=0.0

response = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages = message,
    temperature=temperature,
    max_tokens=max_tokens,
    frequency_penalty=frequency_penalty
)
print(response.choices[0].message)

