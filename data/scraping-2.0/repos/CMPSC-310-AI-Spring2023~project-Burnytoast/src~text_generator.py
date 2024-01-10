import openai

openai.organization = 'org-tiQjuNLO4GEneutRBVq6wVx6'
openai.api_key = ''  # Add your API key here

def generate_text(prompt, model, temperature=0.5, max_tokens=712):
    response = openai.Completion.create(
      engine=model,
      prompt=prompt,
      temperature=temperature,
      max_tokens=max_tokens,
      n=1,
      stop=None,
      timeout=10
    )

    if response.choices:
        return response.choices[0].text.strip()
    else:
        return ""

prompt = "Congratulations on your election victory! As a newly elected politician, you have a unique opportunity to make a difference in the lives of your constituents. Write a speech to inspire them and outline your vision for the future."
model = "text-davinci-002"

generated_text = generate_text(prompt, model, temperature=0.5, max_tokens=712)
print(generated_text)

with open('politician/speech.txt', 'w') as f:
    f.write(generated_text)