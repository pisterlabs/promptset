import gradio as gr
import os
os.environ["OPENAI_API_KEY"] = "open api key 입력"  
from langchain import OpenAI, ConversationChain
import requests
import json

llm = OpenAI(temperature=0.95) # chatgpt 불러오기
conversation = ConversationChain(llm=llm, verbose=False) # 이전 대화를 통해 답변할수 있는 conversationchain 설정


# 아래 window 함수를 통해서 gradio상의 주소?key=1234-1234 라고 한다면 1234-1234를 가져오는 역할을 수행
get_window_url_params = """
    function(text_input, url_params) {
        console.log(text_input, url_params);
        const params = new URLSearchParams(window.location.search);
        url_params = Object.fromEntries(params);
        return [text_input, url_params];
        }
    """

api_request_env = "https://api.aifactory.space/task/checkServiceRequest"

def sendRequestForService(key):
  res = requests.post(api_request_env, json= {'service': 'gui', 'key': key})
  return res

def parse_URL_params(text, url_params):
    return [text, url_params]


def chat(message, history):  
    error = ""
    try:      
        key = url_params['key']   
        print(key)     
        res = sendRequestForService(key)      
        json_data = json.loads(res.text)      
        if(json_data['ct'] == 1) : # 오류 발생시
            return  ["",  json_data['message'], url_params]
        
        # execute predict function 

    except Exception as e:
        print("error")
        print(str(e))

    history = history or []  
    response = conversation.predict(input=message) # conversationchain에 사용자의 질문을 전달
    history.append((message, response)) 
    # conversation chain의 응답을 두개 return 하는데 하나는 gradio chatbot에 전달  
    # 다른 하나는 state에 전달하여 대화 내역을 기록하는 용도

    return history, history 


url_params = gr.JSON({}, visible=True, label="URL Params") # key 값을 url로 부터 가져올때 할당하는 변수
with gr.Blocks() as demo:

    state = gr.State([])
    gen_btn = gr.Button(value = '초기화') 
    ga = gr.Textbox(visible=False)

    # gen_btn 클릭시 window_url_params를 통한 key 정보를 가져오는 기능
    gen_btn.click(fn=parse_URL_params, inputs=[ga, url_params], outputs=[ga, url_params], _js=get_window_url_params)

    # url_params.render() # 실제 오는지 확인용

    chatbot = gr.Chatbot()
    with gr.Row():
        with gr.Column():
            txt = gr.Textbox(
                show_label=False,
                placeholder="Enter text and press enter"
            )
            
        with gr.Column():
            submit_btn = gr.Button(value="Send", variant="secondary")


    # chat 함수 호출하는 submit_btn 과 txt 입력박스에서 enter를 통해 전송하는 txt.submit
    submit_btn.click(chat, 
            inputs=[txt, state], 
            outputs=[chatbot, state])
    txt.submit(chat, [txt, state], [chatbot, state])


if __name__=='__main__':
    demo.launch()
