## Interactive Story Writer

import gradio as gr
import openai
import json

with open('config.json') as f:
    config = json.load(f)

# Get config
openai.api_key = config['CHAT_GPT_API_KEY']
story_parts = config['STORY_PARTS']
story_ans_choices = config['STORY_ANS_CHOICES']
model = "gpt-3.5-turbo"
temperature = 0.7

def get_genre_prompt():
    genres = ["Mystery", "Fantasy", "Science Fiction", "Romance", "Adventure"]
    return "Select a story genre: " + ', '.join(genres)

# Define context
context = f"You are an interactive story writer. Interactive in the sense that you'd start by asking the user for a story genre. The user would input the genre and you'd write the first part of the story. There would be a total of {story_parts} parts to the story. After writing the first part, you'd give the user {story_ans_choices} options that would be the decisions of the main character to proceed with the story. The user would then select an option and then you'd create the next part of the story according to that action. Finally, the story should conclude after part {story_parts}."

# Define message history
message_history = [{'role': 'system', 'content': get_genre_prompt()},
                   {'role': 'user', 'content': context + 'If you understand say "Ok"'},
                   {'role': 'assistant', 'content': 'Ok'},
                   {'role': 'user', 'content': 'So give me story genre to choose from'}]

def predict(input):
    '''
    Generate a response to the user's input
    '''
    global message_history
    global model
    message_history.append({'role': 'user', 'content': input})
    completion = openai.ChatCompletion.create(
        model=model,
        messages=message_history,
        temperature=temperature,
    )
    
    reply_content = completion.choices[0].message.content
    print(reply_content)
    message_history.append({'role': 'assistant', 'content': reply_content}) 
    response = [(message_history[i]["content"], message_history[i+1]["content"]) for i in range(2, len(message_history)-1,2)]

    return response

# Build Gradio UI
with gr.Blocks() as demo:
    chatbot = gr.Chatbot()
    with gr.Row():
        txt = gr.Textbox(show_label=False, placeholder="Enter your message here")
        print("User input: " + txt.value)
        txt.submit(predict, txt, chatbot)
        txt.submit(lambda: "", None, txt) 

# Runner
demo.launch(share=False)