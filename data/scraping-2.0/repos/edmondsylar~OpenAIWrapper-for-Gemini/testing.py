import requests
import autogen
from autogen import OpenAIWrapper

def query_model(prompt):
    response = requests.get(f"http://localhost:12000/ModelCall/{prompt}")
    return response.json()


def UserInteractionTest():
    while True:
        prompt = input("Enter a prompt: ")
        response = query_model(prompt)
        print(response, '\n\n')


# autogen test
def autogenTest():
    config_list = [
        {
        "api_type": "open_ai",
        "api_base": "http://localhost:12000/v1",
        "api_key": "test"
        }
    ]


    client = OpenAIWrapper(config_list=config_list)
    response = client.create(messages=[{"role": "user", "content": "2+2="}])
    print(client.extract_text_or_completion_object(response))


autogenTest()

    

    
