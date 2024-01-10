import pandas as pd
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from langchain.prompts import PromptTemplate

def get_pipeline(model_id):
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_id, load_in_8bit=True, max_length=2048)
    pipe = pipeline(
        "text2text-generation",
        model=model, 
        tokenizer=tokenizer
    )
    return pipe

def get_prompt_template():
    pass

def summarize_log(pipe, log):
    #clear cache
    pass


if __name__ == '__main__':
    #load the logs
    logs = pd.read_csv("logs.csv")
    #get the list of servers
    servers = logs["host"].unique()
    logs_by_server = {}
    #for each server
    for server in servers:
        #get the logs of the server
        logs_of_server = logs[logs["host"] == server]
        #save the logs of the server
        logs_by_server[server] = logs_of_server
    
    #use the model to summarize the logs of each server
    pipe = get_pipeline("google/flan-t5-xxl")

    for server in servers:
        logs_of_server = logs_by_server[server]
