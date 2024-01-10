import openai
import os
from dotenv import load_dotenv

load_dotenv()

# Insert your own api_base, version, and key
openai.api_type = "azure"
openai.api_base =  os.getenv("AZURE_OPENAI_ENDPOINT")
openai.api_version =  os.getenv("OPEN_AI_API_VERSION")
openai.api_key = os.getenv("AZURE_OPENAI_API_KEY")

def get_response(content, prompt_postfix):
    import tiktoken

    response_chunck = []
    n = 1000 # max tokens for chuncking
    max_tokens = 500 # max tokens for response

    tokenizer = tiktoken.get_encoding('p50k_base')

    # Generate chunkcs    
    chunks = chunk_generator(content, n, tokenizer)

    # Decode chunk of text
    text_chunks = [tokenizer.decode(chunk) for chunk in chunks]

    # Request api
    for chunk in text_chunks:
        response_chunck.append(request_api(chunk, prompt_postfix, max_tokens))
        #print(chunk)
        #print('>>> synopsis: \n' + synopsis_chunck[-1])

    # response
    response = ' '.join(response_chunck)

    return response

def chunk_generator(text, n, tokenizer):
    tokens = tokenizer.encode(text)
    i = 0
    while i < len(tokens):
        # Find the nearest end of sentence within a range of 0.5 * n and 1.5 * n tokens
        j = min(i + int(1.5 * n), len(tokens))
        while j > i + int(0.5 * n):
            # Decode the tokens and check for full stop or newline
            chunk = tokenizer.decode(tokens[i:j])
            if chunk.endswith(".") or chunk.endswith("\n"):
                break
            j -= 1
        # If no end of sentence found, use n tokens as the chunk size
        if j == i + int(0.5 * n):
            j = min(i + n, len(tokens))
        yield tokens[i:j]
        i = j
        
def request_api(document, prompt_postfix, max_tokens):
    prompt = prompt_postfix.replace('<document>',document)
    #print(f'>>> prompt : {prompt}')

    response = openai.Completion.create(  
        engine="test-model",
        prompt=prompt,
        temperature=0.3,
        max_tokens=max_tokens,
        top_p=1,
        frequency_penalty=1,
        presence_penalty=1,
        stop='###')

    return response['choices'][0]['text']