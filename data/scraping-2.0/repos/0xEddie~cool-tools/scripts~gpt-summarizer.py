import openai
import clipboard
import keyboard
import configparser

# Set up OpenAI API credentials
config = configparser.ConfigParser()
config.read('config.ini')

openai.api_key = config['openai']['api_key']

def submit_clipboard():
    prompt = "Summarize this statement into bullet points:\n'" + clipboard.paste() + "'\n"
    response = openai.Completion.create(
        engine="davinci",
        prompt=prompt,
        max_tokens=1024,
        n=1,
        stop=None,
        temperature=0.5,
    )
    explanation = response.choices[0].text.strip()
    clipboard.copy(explanation)

keyboard.add_hotkey('ctrl+alt+f3', submit_clipboard)

keyboard.wait('ctrl+esc')