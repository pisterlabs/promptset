import os
import openai
import weaviate

openai_key = os.getenv('OPENAI_KEY')
weaviate_key = os.getenv('WEAVIATE_KEY')
weaviate_url = os.getenv('WEAVIATE_URL')
openai.api_key = os.getenv('OPENAI_KEY')

def answer_chat(query: str) -> str: #1/10th of cost compared to anwer, i.e. text-davinci
    res = openai.ChatCompletion.create(
        model = 'gpt-3.5-turbo',
        messages=[
            {'role': 'system', 'content': 'You are a helpful assistant, providing information about our workplace. If you are unsure, please let us know in the answer and refer to the book named personalhåndboka. Your name is Beate. You speak Norwegian, but can answer in other languages when the question is posed in this language.'},
            {'role': 'user', 'content': query}
            ],
        temperature=0,
        max_tokens=400
    )
    return res['choices'][0]['message']['content']

client = weaviate.Client(
    url=weaviate_url,
    auth_client_secret=weaviate.auth.AuthApiKey(api_key=weaviate_key),
    additional_headers={
        "X-OpenAI-Api-Key": openai_key
    }
)

def get_response(query, history):

    limit = 5000
    # get relevant contexts
    res = (
        client.query
        .get("Personalbok", ["content"])
        .with_near_text({"concepts": [query]})
        .with_limit(20)
        .do()
    )
    contexts = [i["content"] for i in res["data"]["Get"]["Personalbok"]][::-1]
    
    qst_string = ""
    counter = 1
    for q in history[-3:]:
       qst_string += "Spørsmål "+ str(counter)+":"+q["question"]
       qst_string += "Svar "+ str(counter)+":"+q["answer"]

    # build our prompt with the retrieved contexts included
    prompt_start = (
        "Svar på spørsmålet basert på tidligere spørsmål og svar eller kontekst under."
        "Tidligere spørsmål og svar:"+qst_string+
        "Kontekst:\n"
    )
    prompt_end = (
        f"\n\nSpørsmål: {query}\nSvar:"
    )
    # append contexts until hitting limit
    for i in range(1, len(contexts)):
        if len("\n\n---\n\n".join(contexts[:i])) >= limit:
            prompt = (
                prompt_start +
                "\n\n---\n\n".join(contexts[:i-1]) +
                prompt_end
            )
            break
        elif i == len(contexts)-1:
            prompt = (
                prompt_start +
                "\n\n---\n\n".join(contexts) +
                prompt_end
            )
    print(prompt)
    return answer_chat(prompt)