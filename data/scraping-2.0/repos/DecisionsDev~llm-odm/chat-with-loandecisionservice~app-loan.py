import gradio as gr
from langchain.tools import OpenAPISpec, APIOperation
from langchain.chains import OpenAPIEndpointChain, LLMChain
from langchain.requests import RequestsWrapper
import os, yaml, json, time, random,tiktoken, base64
from langchain.llms.openai import OpenAI
from langchain.memory import ChatMessageHistory
import re
from langchain.prompts import PromptTemplate
import ssl
ssl.SSLContext.verify_mode = ssl.VerifyMode.CERT_OPTIONAL
# Check environment variable
ODMRUNTIMEURL="http://localhost:9060/"
showRequest=False
def checkEnvironment():
    if not "ODM_ADMIN_PASSWORD" in os.environ:
        print("#Error you should set the environment variable ODM_ADMIN_PASSWORD that correspond to the ODM Admin user odmAdmin")
        exit(1)
        
    if not "OPENAI_API_KEY" in os.environ:
        print ("#Error you should set the environment variable OPENAI_API_KEY. This variable should contain the OpenAI Key")
        exit(1)
    
    if "ODMRUNTIME_URL" in os.environ:
        global ODMRUNTIMEURL
        ODMRUNTIMEURL=os.environ["ODMRUNTIME_URL"]
        print("Using ODMRUNTIME_URL : ",ODMRUNTIMEURL)
        

def decisionServiceOperation():

    spec = OpenAPISpec.from_url(ODMRUNTIMEURL+"DecisionService/rest/v1/mydeployment/1.0/Miniloan_ServiceRuleset/1.0/OPENAPI?format=YAML")
    operation = APIOperation.from_openapi_spec(spec, '/mydeployment/1.0/Miniloan_ServiceRuleset/1.0', "post")
    return operation

def computeQuestions(operation,llm):
    template = """Below is a service description:

    # {spec}

    # Imagine you're a new user trying to use {operation} through a search bar.
    # What are 10 different things you want to request?
    # Wants/Questions:
    # 1. """

    prompt = PromptTemplate.from_template(template)

    generation_chain = LLMChain(llm=llm, prompt=prompt)

    questions_ = generation_chain.run(spec=operation.to_typescript(), operation="The Decision Service loan service.").split('\n')
    # Strip preceding numeric bullets
    questions = [re.sub(r'^\d+\. ', '', q).strip() for q in questions_]
    print(questions)

def initializeLLMAgent(operation):
    llm = OpenAI() # Load a Language Model
    message = "odmAdmin:"+os.environ["ODM_ADMIN_PASSWORD"]
    message_bytes = message.encode('ascii')
    base64_bytes = base64.b64encode(message_bytes)
    base64_message = base64_bytes.decode('ascii')
    headers = {
        "Authorization": f"Basic "+base64_message
    }
    odm_requests_wrapper=RequestsWrapper(headers=headers)

    #computeQuestions(operation=operation,llm=llm)
    chain = OpenAPIEndpointChain.from_api_operation(
                                                    operation, 
                                                    llm, 
                                                    verify=False,
                                                    requests=odm_requests_wrapper, 
                                                    verbose=True,
                                                    return_intermediate_steps=True)
    return chain




# Verify all is ready to use
print("Verifying the environment")
checkEnvironment()
print("ssssss: ",ODMRUNTIMEURL)
# Read the Swagger API
print ("Loading the API")
operation=decisionServiceOperation()

print("Creating the LLMAgent")
odm_agent = initializeLLMAgent(operation)

def show_text(x):
    global showRequest
    showRequest=x
    return showRequest
def add_text(history, text):
    history = history + [(text, None)]
    return history, ""

def bot(history):
    response = infer(history[-1][0])
    request=""
    if (showRequest): 
        request=f"Loan Request : <span style='color:blue'>{response['intermediate_steps']}</span>\n\n"

    history[-1][1] = f"{request}<span style='color:green'>{response['output']}</span>"
    return history

def infer(question):
    query = question
#    response={'output': {'instructions': 'my name is lolo, i want to do a loan of an amount of 100000 with a credit score of 210 with a duration of 100 is it possible?', 'output': 'Based on the API response, it looks like the loan has been denied. This could be due to a low credit score, insufficient loan amount, or other factors. If you would like to explore other loan options, please contact a loan provider.', 'intermediate_steps': {'request_args': '{"borrower": {"name": "lolo", "creditScore": 210, "yearlyIncome": 0}, "loan": {"amount": 100000, "duration": 100, "yearlyInterestRate": 0, "yearlyRepayment": 0, "approved": false, "messages": []}}', 'response_text': '{"__DecisionID__":"da57adb4-1830-4d20-bc3d-ed48851308340","loan":{"amount":100000,"duration":100,"yearlyInterestRate":0.0,"yearlyRepayment":0,"approved":false,"messages":[]}}'}}}
    response={}
    response['output'] = odm_agent(query)
    return response['output']

css="""
#col-container {max-width: 700px; margin-left: auto; margin-right: auto;}
"""

title = """
<div style="text-align: center;max-width: 700px;">
    <h1>Chat with ODM</h1>
    <p style="text-align: center;">This sample allow to interact with the Loan Decision Service. 
    Here some questions you can ask to the ODM chat system:
    <li>my name is joe, i want to do a loan of an amount of 100000 with a credit score of 210 with a duration of 100 is it possible?</li>
    <li>Will it be possible to contract a loan of an amount of 100000 with a credit score of 210 during 10 year ?</li>
    <li>Will it be possible to contract a loan of an amount of 200000 with a credit score of 210 during 10 year ?</li>
    </p>
 </div>
"""

with gr.Blocks(css=css) as demo:
    with gr.Column(elem_id="col-container"):
        gr.HTML(title)
        
        chatbot = gr.Chatbot([], show_label=True,label="Chatbot Output",elem_id="chatbot",visible=True,color_map=["blue","grey"]).style(height=350)
        question = gr.Textbox(label="Question", placeholder="Type your question and hit Enter ")
        submit_btn = gr.Button("Send message")
        check= gr.Checkbox(label="Display Request", info="Show request call")
        check.change(fn=show_text, inputs=[check], outputs=[])

    question.submit(add_text, [chatbot, question], [chatbot, question]).then(
        bot, chatbot, chatbot
    )
    
    submit_btn.click(add_text, [chatbot, question], [chatbot, question]).then(
        bot, chatbot, chatbot
    )

demo.launch()