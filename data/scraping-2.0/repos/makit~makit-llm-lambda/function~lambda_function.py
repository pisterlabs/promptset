import json
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.llms import GPT4All

template = """You are a Text-Based Fantasy RPG. The player is exploring the dark forest and you are the dungeon master. The player performs the action: {action} What happens next? You are talking to the player."""
prompt = PromptTemplate(template=template, input_variables=["action"])
local_path = ("./ggml-model-gpt4all-falcon-q4_0.bin")
llm = GPT4All(model=local_path, verbose=True)
llm_chain = LLMChain(prompt=prompt, llm=llm)

def handler(event, context):
    body = json.loads(event.get("body", "{}"))
    action = body.get("action")
    
    if action is None:
        return {
            'statusCode': 400,
            'body': json.dumps({'error': 'No action was provided'})
        }

    response  = llm_chain.run(action)

    return {
        'statusCode': 200,
        'body': json.dumps(response)
    }