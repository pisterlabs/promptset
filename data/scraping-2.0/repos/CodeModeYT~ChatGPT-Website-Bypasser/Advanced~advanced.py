import openai
import os
from colorama import Fore
from datetime import datetime

# Remove this if you want to
print(Fore.BLUE + "ChatGPT website bypasser - advanced version!")
print(Fore.RED + 'IMPORTANT: Read through the readme file on GitHub first, otherwise this file will NOT work')
print(Fore.WHITE + "github.com/CodeModeYT/ChatGPT-Website-Bypasser")

def save2txt(response):
    now = datetime.now()
    currenttime = now.strftime("%d.%m.%Y %H.%M.%S")
    filename = f"ChatGPT at {currenttime}"
    results_path = f'D:\Path-to\project\ChatGPT-Website-Bypasser\Advanced\Results\{filename}.txt' #(Replace with your own Path)
    directory = os.path.dirname(results_path)
    with open(results_path, 'w') as f:
        f.write(f"Result from the API: {response}")
    print(f"Sucessfully saved the response as a txt file in the path {results_path}")
    print("Continuing...")
    print("---------")
    generate()

def check(response):
    # Checking if the user wants to save the results as a txt file:
    choice = input("Would you like to save the result in a text file? (y/n)")
    if choice == "y":
        save2txt(response)
    elif choice == "n":
        print("OK, not saving the response. Continuing...")
        print("---------")
        generate()
    else:
        print("Not a valid choice, please either type 'y' or 'n' and confirm with enter!")
        print("---------")
        check(response)

def generate():    # Get the user input
    usrinput = input("Write your question and confirm with enter: ")

    # Set up the OpenAI API client
    openai.api_key = "API-key-here"

    # Set up the model and prompt
    model_engine = "text-davinci-003"
    prompt = usrinput

    # Generate a response
    completion = openai.Completion.create(
        engine=model_engine,
        prompt=prompt,
        max_tokens=1024,
        n=1,
        stop=None,
        temperature=0.5,
    )
    response = completion.choices[0].text
    # Printing it out
    print("Response from API:")
    print(response)
    print("---------")
    check(response)
    

#Running the function
generate()