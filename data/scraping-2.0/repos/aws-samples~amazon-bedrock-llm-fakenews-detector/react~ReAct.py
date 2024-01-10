import jsonlines
import boto3
from langchain.agents import load_tools
from langchain.agents import initialize_agent
from langchain.agents import AgentType
from langchain.llms.bedrock import Bedrock
from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv()) # read local .env file

def get_llm(model_id='amazon.titan-tg1-large'):    
    boto3_bedrock = boto3.client('bedrock-runtime', region_name='us-west-2')
    llm = Bedrock(model_id=model_id, client=boto3_bedrock, credentials_profile_name="genai")
    return llm  

def generate_and_print(llm, q, label):
    print(f'Inside generate_and_print: q = {q}')
    tools = load_tools(["wikipedia"], llm=llm)
    agent = initialize_agent(tools, llm, 
                             agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, 
                             verbose=True,
                             handle_parsing_errors=True,
                             agent_kwargs={})

    input = """Here is a statement:
    {statement}
    Is this statement is correct? You can use tools to find information if needed.
    The final response is FALSE if the statement is FALSE. Otherwise, TRUE."""

    answer = agent.run(input.format(statement=q))
    
    return answer

def read_questions(llm):
    file='./knowledge_qa_test.jsonl'
    with jsonlines.open(file,'r') as json_f:
        for data in json_f:
            prompt = data.get("prompt", "")
            response = data.get("response", "")
            claims = data.get("claims", [])
            label = data.get("label", "")
            entry_point = data.get("entry_point", "")

            print("Prompt:", prompt)
            print("Response:", response)
            print("Claims:", claims)
            print("label:", label)
            print("entry_point:", entry_point)
            print("\n")
            generate_and_print(llm, response, label)

def main():

    llm = get_llm(model_id="anthropic.claude-v2")

    read_questions(llm)

if __name__ == "__main__":
    main()