from dotenv import load_dotenv
load_dotenv()
import os
api_key = os.getenv("API_KEY")

## 

from openai import OpenAI
import gradio as gr

client = OpenAI(
   
    api_key=api_key,
)

user_content = input("Ask me any question:  ")
def gpt_output(user_content):
    
    conversation = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": user_content}
        # Add more user and assistant messages as needed
    ]

    response = client.chat.completions.create(
         model="gpt-3.5-turbo",
         messages=conversation,         
         temperature=1,
         max_tokens=256,
         top_p=1,
         frequency_penalty=0,
         presence_penalty=0
   
)
 
    # model_reply = response.choices[0].message.content
    print("AI Response:  ", response.choices[0].message.content)
    # return("AI Response:  ", response.choices[0].message.content)
    # return f"AI Response: {model_reply}"


gpt_output(user_content)
# while True:
#     query= user_content
#     gpt_output(query)

block = gr.Blocks()

with block:
    gr.Markdown("""<h1><center>AGI AI Assistant</center></h1>""")
    chatbot = gr.Chatbot()
    message = gr.Textbox(placeholder="Enter your query")
    state = gr.State()
    submit = gr.Button("Send")
    submit.click(gpt_output)
    # submit.click(gpt_output, inputs=[message, state], outputs=[chatbot, state])

block.launch(debug=True, share=True)
