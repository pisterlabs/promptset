import os
from openai import AzureOpenAI
from dotenv import load_dotenv

def assist4():
    load_dotenv()
    client = AzureOpenAI(
        api_key=os.getenv("AZURE_OPENAI_KEY"),  
        api_version="2023-10-01-preview",
        azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        )
    deployment_name='GPT4' #This will correspond to the custom name you chose for your deployment when you deployed a model. 
    """ response = client.chat.completions.create(
        model="GPT4", # model = "deployment_name".
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Does Azure OpenAI support customer managed keys?"},
            {"role": "assistant", "content": "Yes, customer managed keys are supported by Azure OpenAI."},
            {"role": "user", "content": "Do other Azure AI services support this too?"}
        ]
    )

    print(response.choices[0].message.content) """

    response = client.chat.completions.create(
        model="GPT4", # model = "deployment_name".
        messages=[
            {"role": "system", "content": "Vous êtes un assistant d’un centre d’appel voulant aider les utilisateurs."},
            {"role": "user", "content": "Je n’arrive pas à imprimer"},
            {"role": "assistant", "content": "Vérifier si votre imprimante est bien configurée dans le panneau de configuration"},
            {"role": "user", "content": "Comment changer mon mot de passe ?"}
        ]
    )
    print(response.choices[0].message.content)


def get_completion(prompt, model="gpt-3.5-turbo"):
    messages = [{"role": "user", "content": prompt}]
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=0, # this is the degree of randomness of the model's output
    )
    return response.choices[0].message["content"]

if __name__ == "__main__":
    #assist4()
    load_dotenv()
    client = AzureOpenAI(
        api_key=os.getenv("AZURE_OPENAI_KEY"),  
        api_version="2023-10-01-preview",
        azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        )
    deployment_name="gpt35turbo"
    # Send a completion call to generate an answer
    text = f"""
    You should express what you want a model to do by \ 
    providing instructions that are as clear and \ 
    specific as you can possibly make them. \ 
    This will guide the model towards the desired output, \ 
    and reduce the chances of receiving irrelevant \ 
    or incorrect responses. Don't confuse writing a \ 
    clear prompt with writing a short prompt. \ 
    In many cases, longer prompts provide more clarity \ 
    and context for the model, which can lead to \ 
    more detailed and relevant outputs.
    """
    prompt = f"""
    Summarize the text delimited by triple backticks \ 
    into a single sentence.
    ```{text}```
    """
    
    #
    ### COMPLETION API
    #
    #start_phrase = prompt
    #response = client.completions.create(model=deployment_name, prompt=start_phrase, max_tokens=300)
    #print(response.choices[0].text)

    deployment_name='GPT4' #This will correspond to the custom name you chose for your deployment when you deployed a model. 
    response = client.chat.completions.create(
        model="GPT4", # model = "deployment_name".
        messages=[
            {"role": "system", "content": "You are an assistant designed to summarize text"},
            
            {"role": "user", "content": prompt}
        ]
    )
    print(response.model_dump_json(indent=2))
    print(response.choices[0].message.content)