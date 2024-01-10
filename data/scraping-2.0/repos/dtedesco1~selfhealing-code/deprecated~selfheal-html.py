import openai
import os
import webbrowser

# Set the OpenAI API key
openai.api_key = open("key.txt", "r").read().strip("\n")

# Define the function to generate the webpage
def generate_webpage(user_input):
    # Send the user input to GPT-3 to generate the HTML
    response = openai.Completion.create(
        engine="davinci",
        prompt=f"Create an HTML webpage that does the following:\n{user_input}\n",
        max_tokens=1024,
        n=1,
        stop=None,
        temperature=0.5,
    )
    # Extract the generated HTML from the API response
    generated_html = response.choices[0].text.strip()
    return generated_html

# Define the function to fix errors using GPT-3
def fix_error(error_msg):
    # Send the error message to GPT-3
    response = openai.Completion.create(
        engine="davinci-codex",
        prompt=f"Fix the following HTML error:\n{error_msg}\n",
        max_tokens=1024,
        n=1,
        stop=None,
        temperature=0.5,
    )
    solution = response.choices[0].text.strip()
    # Modify the HTML to fix the error
    global webpage_html
    webpage_html = webpage_html.replace(error_msg, solution)

# Define the main function to generate the webpage
def main():
    user_input = input("Please enter what you would like the webpage to do: ")
    global webpage_html
    webpage_html = generate_webpage(user_input)
    while True:
        try:
            # Open the generated webpage in the default web browser
            with open("generated_webpage.html", "w") as f:
                f.write(webpage_html)
            webbrowser.open("file://generated_webpage.html")
        except Exception as e:
            error_msg = str(e)
            fix_error(error_msg)
        else:
            break

# Call the main function to start the process
main()