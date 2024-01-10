import os
import logging
import openai
import pyfiglet
import pwinput

ascii_banner = pyfiglet.figlet_format("ChatGPT in CLI!")
print(ascii_banner)

if openai.api_key:
    logging.info(f"Proceeding since environment variable OPENAI_API_KEY exists {openai.api_key=}")
    pass
else:
    openai.api_key = pwinput.pwinput(prompt='Please input your OpenAI API key:', mask='*')
    pass

def chatgpt():
    chatgpt_ques = input("Ask Me 🤖 \n")
    response = openai.Completion.create(
        model="text-davinci-003",
        prompt=chatgpt_ques,
        max_tokens=3000,
        temperature=0.7
    )
    """Respond to question with ChatGPT's response."""
    print(response["choices"][0]["text"])

    return str(response)

chatgpt();