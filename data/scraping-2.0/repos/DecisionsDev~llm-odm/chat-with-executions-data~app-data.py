import gradio as gr
from langchain.tools import OpenAPISpec, APIOperation
from langchain.chains import OpenAPIEndpointChain, LLMChain
from langchain.llms import OpenAI
from pyspark.sql import SparkSession
from langchain.agents import create_spark_dataframe_agent
import os, yaml, json, time, random,tiktoken, base64
from langchain.llms.openai import OpenAI
from langchain.prompts import PromptTemplate


showRequest=False
def checkEnvironment():
    if not "OPENAI_API_KEY" in os.environ:
        print ("#Error you should set the environment variable OPENAI_API_KEY. This variable should contain the OpenAI Key")
        exit(1)
        

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

def initializeLLMAgent():
    agent = create_spark_dataframe_agent(llm=OpenAI(temperature=0), df=df, verbose=True)
    return agent


# Verify all is ready to use
print("Verifying the environment")
checkEnvironment()


print("Creating the LLMAgent")
spark = SparkSession.builder.getOrCreate()
#csv_file_path = "credit-risk-100.csv"
#csv_file_path = "loanvalidation-decisions-10K.csv"
csv_file_path = "loanvalidation-1K.csv"
df = spark.read.csv(csv_file_path, header=True, inferSchema=True)

odm_agent = initializeLLMAgent()

def display_df():
  df_images = df.head(10)
  
  return df_images

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
#col-container { margin-left: auto; margin-right: auto;}
"""

title = """
<div style="text-align: center;">
    <h1>Decision Automation Chat</h1>
    <p style="text-align: center;">This sample allow to interact with the Loan Decision Service. 
    Here some questions you can ask to the Decision chat system:
    <li>As an end user, please describe the format of the dataset based on the column name.</li>
    <li>What is the number of rejected loan applications?</li>
    <li>What is the sum of the accepted loan amounts?</li>
    </p>
 </div>
"""


with gr.Blocks(css=css) as demo:
    with gr.Column(elem_id="col-container"):
        gr.HTML(title)
        with gr.Row():
            out_dataframe = gr.Dataframe(wrap=True,value=display_df, max_rows=10,headers=df.columns, overflow_row_behaviour= "paginate", interactive=False,default=display_df)
 #            table = gr.Dataframe(headers=["Phase", "Activity", "Start date", "End date"], col_count=4, default=process_csv_text(default_csv))
        b1 = gr.Button("Get Initial dataframe")
        b1.click(fn=display_df, outputs=out_dataframe, api_name="initial_dataframe") 
      
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