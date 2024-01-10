from openai import OpenAI
import tiktoken

client = OpenAI()

def get_chat_response(messages, gpt_version, openai_apikey):
    openai_api_key = openai_apikey

    response = client.chat.completions.create(
        messages = messages,
        model = gpt_version,
        temperature = 0.7,
        top_p = 0.5,
        #n = 1,
        stop = None,
        frequency_penalty = 0.0,
        presence_penalty = 0.0,
        max_tokens = 1200
    )

    return response.model_dump()



def num_tokens_from_string (string: str):
    encoding = tiktoken.encoding_for_model("gpt-4-0613") 
    num_tokens = len(encoding.encode(string))
    return num_tokens

def call_open_ai(openai_apikey):
    payload = []
    modelo ="gpt-4-1106-preview"
    contagem = []
    count = 1

        

    full_prompt = "escreva seu prompt auqi"
        
    messages = [{"role": "user", "content": full_prompt}]    
        
    payload= get_chat_response(messages, modelo, openai_apikey)

    modelo_final = payload.model
    tokens_finais = payload.usage
    tokens_finais["modelo"] = modelo_final

    resposta_texto = payload.choice[0].message["content"]



    contagem.append(tokens_finais)
    count += 1

    return resposta_texto, contagem