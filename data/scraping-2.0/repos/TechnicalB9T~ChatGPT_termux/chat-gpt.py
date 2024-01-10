import os
import openai
import pyfiglet
from termcolor import colored

path = os.getcwd()
file_path = f'{path}/api.txt'

if os.path.exists(file_path):
    result = pyfiglet.figlet_format("ChatGPT")
    print(colored(result,'green'))


    file = open(file_path,'r')
    data = file.read()
    openai.api_key = data
    file.close()
    prompt = input(colored("enter the querry :",'blue'))
    for i in range (4):
        print(colored(">"*i,'blue'))

    response = openai.Completion.create(
    engine='text-davinci-003',  # Use 'text-davinci-003' for gpt-3.5-turbo model
    prompt=prompt,
    max_tokens= 500 )
    generated_text = response.choices[0].text.strip()
    print(generated_text)



else:
    with open("api.txt","w") as file:
        file.write(input("Enter your api key: "))

    file.close()

