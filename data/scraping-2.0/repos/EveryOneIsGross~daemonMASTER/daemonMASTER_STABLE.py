import openai
import dotenv
import os
import json
from rake_nltk import Rake
from textblob import TextBlob
import gradio as gr
from gradio.components import *
from nltk.corpus import stopwords
from nltk.corpus import wordnet
from openai.embeddings_utils import get_embedding, cosine_similarity
import pandas as pd
from gradio import Interface, components

#download wordnet
#import nltk
#nltk.download('wordnet')

class daemonMASTER:
    def __init__(self, name):
        dotenv.load_dotenv()
        openai.api_key = os.getenv('OPENAI_API_KEY')
        self.name = name
        self.system_prompts = {
            'chat': "",
            'instruct': "",
            'summarise': "Summarise the following:",
        }
        self.previous_responses = {}
        self.rake = Rake(stopwords=stopwords.words('english'))

    def chat(self, message):
        # Append the user's message to the full conversation context
        self.previous_responses.setdefault('chat_context', []).append({"role": "user", "content": message})
        conversation = [{"role": "system", "content": f"act only in character as {self.name}."}] + self.previous_responses.get('chat_context', [])
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo-16k",
            messages=conversation,
            max_tokens=200,
            temperature=0.6,
        )
        agent_output = response.choices[0].message.content.strip()
        self.save_output("chat", message, agent_output)
        # Append the assistant's response to the full conversation context
        self.previous_responses['chat_context'].append({"role": "assistant", "content": agent_output})
        # Append the assistant's response to the list of responses to be displayed
        self.previous_responses.setdefault('chat', []).append(agent_output)



    def execute_command(self, command, text):
        if command in ['instruct', 'summarise']:
            prompt = self.previous_responses.get(command, "") + " " + self.system_prompts[command] + " " + text
            response = openai.Completion.create(
                engine="text-davinci-003",
                prompt=prompt,
                max_tokens=150 if command == 'instruct' else 300,
                temperature=0.1
            )
            agent_output = response.choices[0].text.strip()
        elif command == 'sentiment':
            blob = TextBlob(text)
            agent_output = blob.sentiment.polarity
        elif command == 'keywords':
            synonyms = [lemma.name() for synset in wordnet.synsets(self.name) for lemma in synset.lemmas()]
            text_with_synonyms = text + ' ' + ' '.join(synonyms)
            self.rake.extract_keywords_from_text(text_with_synonyms)
            agent_output = self.rake.get_ranked_phrases()
        elif command == 'search':
            agent_output = self.search_chat_history(text)
        else:
            agent_output = "Unknown command"
        self.save_output(command, text, agent_output)
        self.previous_responses[command] = agent_output

    def search_chat_history(self, search_term):
        filename = f"{self.name}_chat.json"
        if os.path.isfile(filename):
            with open(filename, "r") as file:
                data = json.load(file)
            df = pd.DataFrame(data)
            df['embedding'] = df['user_input'].apply(lambda x: get_embedding(x, engine="text-embedding-ada-002"))
            return self.search(df, search_term)
        else:
            return []

    def search(self, df, text, n=3):
        text_embedding = get_embedding(text, engine="text-embedding-ada-002")
        df["similarity"] = df.embedding.apply(lambda x: cosine_similarity(x, text_embedding))
        results = (
            df.sort_values("similarity", ascending=False)
            .head(n)
            .user_input.to_list()
        )
        return results

    def get_last_responses(self, n):
        filename = f"{self.name}_chat.json"
        if os.path.isfile(filename):
            with open(filename, "r") as file:
                data = json.load(file)
            return data[-n:]
        else:
            return []

    def save_output(self, command, user_input, agent_output):
        filename = f"{self.name}_{command}.json"
        data = {
            "user_input": user_input,
            "agent_output": agent_output
        }
        if os.path.isfile(filename):
            with open(filename, "r") as file:
                existing_data = json.load(file)
            if isinstance(existing_data, list):
                existing_data.append(data)
            else:
                existing_data = [existing_data, data]
            with open(filename, "w") as file:
                json.dump(existing_data, file)
        else:
            with open(filename, "w") as file:
                json.dump([data], file)


def interact(bot_name="", chat_prompt="", instruct_prompt="", command="", text=""):
    current_bot = daemonMASTER(bot_name)
    current_bot.system_prompts['chat'] = chat_prompt
    current_bot.system_prompts['instruct'] = instruct_prompt
    gifs = {'chat': 'clippy/clippyIDLE.gif', 'instruct': 'clippy/clippyWEIRD.gif', 'sentiment': 'clippy/clippySHAME.gif', 'summarise': 'clippy/clippyBOX.gif', 'keywords': 'clippy/clippyTHINK.gif', 'last_responses': 'clippy/clippyDISSOLVE.gif', 'search': 'clippy/clippySUMMARISE.gif'}
    image_path = gifs[command]
    if command == 'chat':
        current_bot.chat(text)
    elif command == 'last_responses':
        responses = current_bot.get_last_responses(4)
        return responses, image_path
    else:
        current_bot.execute_command(command, text)
    return current_bot.previous_responses.get(command, ""), image_path

css = """
:root {
  --background-fill-secondary: white !important;
  --shadow-drop-lg: white !important;
  --block-label-border-width: 0px; */
  --block-label-text-color: white */
  --block-label-margin: 0; */
  --block-label-padding: var(--spacing-sm) var(--spacing-lg); */
  --block-label-radius: calc(var(--radius-lg) - 1px) 0 calc(var(--radius-lg) - 0px) 0; */
  --block-label-right-radius: 0 calc(var(--radius-lg) - 0px) 0 calc(var(--radius-lg) - 0px); */
  --block-label-text-size: var(--text-md); */
  --block-label-text-weight: 0;
}
.hide-label .gradio-block-label {display: none;}
.hide-icon .gradio-image-icon {display: none;}

.gradio-input-section {
  --background-fill-secondary: white;
}

.gradio-content {
    background-color: white;
}

.gradio-input-section, .gradio-output-section {
    background-color: white;
    box-shadow: none;
}

.gradio-input, .gradio-output {
    border: none;
    box-shadow: none;
}

body, label, .gradio-textbox textarea {
    font-family: 'Comic Sans MS', 'Comic Sans';
    font-weight: bold;
}
"""

bot_name_input = components.Textbox(lines=1, label="Enter bot name")
chat_prompt_input = components.Textbox(lines=3, label="Enter bot's behavior")
instruct_prompt_input = components.Textbox(lines=3, label="Enter instruction prompt")

commands = ['chat', 'instruct', 'sentiment', 'summarise', 'keywords', 'last_responses', 'search']

command_dropdown = components.Dropdown(commands, label="Select mode")

text_input = components.Textbox(lines=5, label="Enter input")

output_text = components.Textbox(label="Output")
output_image = components.Image(type='filepath', label=" ")

iface = Interface(fn=interact, css=css,
                  inputs=[bot_name_input, chat_prompt_input, instruct_prompt_input, command_dropdown, text_input], 
                  outputs=[output_text, output_image],
                  theme=gr.themes.Monochrome())

iface.launch(share=True)
