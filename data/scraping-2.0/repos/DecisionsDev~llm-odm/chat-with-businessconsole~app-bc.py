import gradio as gr
from langchain.agents.agent_toolkits.openapi.spec import reduce_openapi_spec
from langchain.requests import RequestsWrapper
import os, yaml, json, time, random,tiktoken, base64
from langchain.llms.openai import OpenAI
from langchain.agents.agent_toolkits.openapi import planner


# Check environment variable
def checkEnvironment():
    if not "ODM_ADMIN_PASSWORD" in os.environ:
        print("#Error you should set the environment variable ODM_ADMIN_PASSWORD that correspond to the ODM Admin user odmAdmin")
        exit(1)
        
    if not "OPENAI_API_KEY" in os.environ:
        print ("#Error you should set the environment variable OPENAI_API_KEY. This variable should contain the OpenAI Key")
        exit(1)

def loadDecisionCenterAPI():
    with open("dc-swagger-api.yaml") as f:
        raw_decisioncenter_api_spec = yaml.load(f, Loader=yaml.Loader)
    decisioncenter_api_spec = reduce_openapi_spec(raw_decisioncenter_api_spec)

    # Extract informations about the Swagger ODM API
    endpoints = [
        (route, operation)
        for route, operations in raw_decisioncenter_api_spec["paths"].items()
        for operation in operations
        if operation in ["get", "post"]
    ]
    len(endpoints)    
    print (len(endpoints))
    enc = tiktoken.encoding_for_model('text-davinci-003')
    def count_tokens(s): return len(enc.encode(s))
    print ("Tokens number for the DC API : ",count_tokens(yaml.dump(raw_decisioncenter_api_spec)))
    return decisioncenter_api_spec

def initializeLLMAgent(decisioncenter_api_spec):
    llm = OpenAI(model_name='gpt-4') # Load a Language Model
    message = "odmAdmin:"+os.environ["ODM_ADMIN_PASSWORD"]
    message_bytes = message.encode('ascii')
    base64_bytes = base64.b64encode(message_bytes)
    base64_message = base64_bytes.decode('ascii')
    headers = {
        "Authorization": f"Basic "+base64_message
    }
    odm_requests_wrapper=RequestsWrapper(headers=headers)

    odm_dc_agent = planner.create_openapi_agent(decisioncenter_api_spec, odm_requests_wrapper, llm)
    return odm_dc_agent


# Verify all is ready to use
print("Verifying the environment")
checkEnvironment()

# Read the Swagger API
print ("Loading the API")
dc_api_spec=loadDecisionCenterAPI()

print("Creating the LLMAgent")
odm_dc_agent = initializeLLMAgent(dc_api_spec)


def add_text(history, text):
    history = history + [(text, None)]
    return history, ""

def bot(history):
    response = infer(history[-1][0])
    history[-1][1] = response['result']
    return history

def infer(question):
    
    query = question
#    result = qa({"query": query})

    response={}
    #response['result'] = random.choice(["How are you?", "I love you", "I'm very hungry"])
    response['result'] = odm_dc_agent.run(query)
    return response

css="""
#col-container {max-width: 700px; auto; margin-right: auto;}


    #resources{
      background-color: rgb(30, 58, 75);
      margin: 1000px 10 10 10;

    }
    #resources h3{
      color: white;
      font-size: 2rem;
      font-weight: lighter;
      margin: 10px;
      text-align: left;
    }

  
"""

title = """
        <div id="resources">
        <h3>&nbsp;Business Console Assistant</h3>
        </div>

"""

with gr.Blocks(css=css) as demo:
    with gr.Column(elem_id="col-container"):
        gr.HTML(title)
 
#        chatbot = gr.Chatbot([], elem_id="chatbot").style(height=350)
        chatbot = gr.Chatbot([], show_label=True,label="Chatbot Output",elem_id="chatbot",visible=True,show_copy_button=True,avatar_images=( "./images/user.png", "./images/support.png"))
        question = gr.Textbox(label="Question", placeholder="Type your question and hit Enter ")
        submit_btn = gr.Button("Send message")
    #load_pdf.click(loading_pdf, None, langchain_status, queue=False)    
    # repo_id.change(pdf_changes, inputs=[pdf_doc, repo_id], outputs=[langchain_status], queue=False)
    # load_pdf.click(pdf_changes, inputs=[pdf_doc, repo_id], outputs=[langchain_status], queue=False)
    question.submit(add_text, [chatbot, question], [chatbot, question]).then(
        bot, chatbot, chatbot
    )
    submit_btn.click(add_text, [chatbot, question], [chatbot, question]).then(
        bot, chatbot, chatbot
    )

demo.launch()