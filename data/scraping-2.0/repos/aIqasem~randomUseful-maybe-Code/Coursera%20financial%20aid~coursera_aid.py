import openai
import pyperclip

openai.api_key = "Create your free api key from https://platform.openai.com/api-keys "
model_engine = "text-davinci-003"

completion = openai.Completion.create(
    engine=model_engine,
    prompt=f"""In at least 170 words, answer "How will your selected course help with your goals?" to get course financial aid on Coursera
The course name is '{pyperclip.paste().strip()}'""",
    temperature=0.5,
    max_tokens=1024,
)

response = completion.choices[0].text
print(f"\n  Copied the following response:\n\n{response.strip()}")
pyperclip.copy(response.strip())
