
import os 
import openai 
import gradio 
openai.api_key = ""
prompt = "Enter Your Query Here"
def api_calling(prompt): 
    completions = openai.Completion.create( 
        engine="text-davinci-003", 
        prompt=prompt, 
        max_tokens=1024, 
        n=1, 
        stop=None, 
        temperature=0.5, 
    ) 
    message = completions.choices[0].text 
    return message 
def message_and_history(input, history): 
    history = history or [] 
    s = list(sum(history, ())) 
    s.append(input) 
    inp = ' '.join(s) 
    output = api_calling(inp) 
    history.append((input, output)) 
    return history, history 
block = gradio.Blocks(theme=gradio.themes.Monochrome()) 
with block: 
    gradio.Markdown("""<h1><center>ChatGPT  
    ChatBot with Gradio and OpenAI</center></h1> 
    """) 
    chatbot = gradio.Chatbot() 
    message = gradio.Textbox(placeholder=prompt) 
    state = gradio.State() 
    submit = gradio.Button("SEND") 
    submit.click(message_and_history,  
                 inputs=[message, state],  
                 outputs=[chatbot, state]) 
block.launch(debug = True)