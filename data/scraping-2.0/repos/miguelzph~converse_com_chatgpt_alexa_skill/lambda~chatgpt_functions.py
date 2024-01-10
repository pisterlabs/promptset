import openai
import time


def add_prompt_message(content_input, role="user"):

    dict_to_add = {"role": role, "content": content_input}
    
    return dict_to_add


def streaming_response_from_chatgpt(messages: list, start_time, openapi_key: str, openai_model: str, max_tokens=3000):
    
    openai.api_key = openapi_key

    current_sentence = ""
    
    for resp in openai.ChatCompletion.create(
                        model=openai_model,
                        messages=messages,
                        max_tokens=max_tokens,
                        temperature=0.5,
                        stream=True,
                        request_timeout=5 # é um parametro não documentado, porém não tinha conseguido aplicar o timeout de outra forma
                        ):

        
        try:
            partial_content = resp['choices'][0]['delta']['content']
        except KeyError: # no content response (poucos erros)
            continue

        current_sentence += partial_content

        if ('\n' in partial_content) or (('.' in partial_content) and (len(current_sentence) > 5)): 

            yield current_sentence
            current_sentence = ""

        elif ((time.time() - start_time) > 7.6): # limite de 8 segundos
            yield False
            
    yield current_sentence
            