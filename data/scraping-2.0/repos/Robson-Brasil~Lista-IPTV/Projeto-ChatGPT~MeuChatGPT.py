import openai
openai.api_key = "org-GUyNsYnKz5gwEQaT5vdfjuds"
def get_answer(prompt):
    completions = openai.Completion.create(
        engine="davinci", prompt=prompt, max_tokens=1024, n=1,stop=None,temperature=0.7,)
    message = completions.choices[0].text
    return message.strip()
prompt = "Qual a capital do Brasil?"
answer = get_answer(prompt)
print(answer)