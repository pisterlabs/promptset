import gradio as gr
import openai
 
openai.api_key = 'YOUR API KEY HERE'
 
def answer(state, state_chatbot, text):
    messages = state + [{
        'role': 'user',
        'content': text
    }]
 
    res = openai.ChatCompletion.create(
        model='gpt-3.5-turbo',
        messages=messages
    )
 
    msg = res['choices'][0]['message']['content']
 
    new_state = [{
        'role': 'user',
        'content': text
    }, {
        'role': 'assistant',
        'content': msg
    }]
 
    state = state + new_state
    state_chatbot = state_chatbot + [(text, msg)]
 
    print(state)
 
    return state, state_chatbot, state_chatbot
 
 
with gr.Blocks(css='#chatbot .overflow-y-auto{height:750px}') as demo:
    state = gr.State([{
        'role': 'system',
        'content': 'You are a helpful assistant.'
    }])
    state_chatbot = gr.State([])
 
    with gr.Row():
        gr.HTML("""<div style="text-align: center; max-width: 500px; margin: 0 auto;">
            <div>
                <h1>Yunwoong's ChatGPT-3.5</h1>
            </div>
            <p style="margin-bottom: 10px; font-size: 94%">
                Blog <a href="https://yunwoong.tistory.com/">Be Original</a>
            </p>
        </div>""")
 
    with gr.Row():
        chatbot = gr.Chatbot(elem_id='chatbot')
 
    with gr.Row():
        txt = gr.Textbox(show_label=False, placeholder='Send a message...').style(container=False)
 
    txt.submit(answer, [state, state_chatbot, txt], [state, state_chatbot, chatbot])
    txt.submit(lambda: '', None, txt)
 
demo.launch(debug=True, share=True)