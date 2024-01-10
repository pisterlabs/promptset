from langchain.llms import VertexAI
import requests
import json

import numpy as np
import subprocess
import time

def translate_to_english(txt):
    # random delay not to upset the server
    #time.sleep(np.random.random()*5)
    
    txt = txt.replace("'","''").replace("\n","")
    command = f"curl --header 'Accept: text/json' --user-agent 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10.8; rv:21.0) Gecko/20100101 Firefox/21.0' --data-urlencode 'client=gtx' --data-urlencode 'sl=auto' --data-urlencode 'tl=en' --data-urlencode 'dt=at' --data-urlencode 'q={txt}' -sL http://translate.googleapis.com/translate_a/single | jq -r '.[5][][2][0][0]'"
#     print(command)
    out = subprocess.run(command, shell=True, stdout=subprocess.PIPE)
    
    return out.stdout.decode("utf-8")#.replace("\n","")


llm = VertexAI(
  model_name='text-bison',
  max_output_tokens=1024,
  temperature=0.0,
  top_p=0.2,
  top_k=20,
  verbose=True
)


def build_prompt(chat_history, chunks):
    hist = "\n".join([f"{m['name'].upper()}: {m['text']}" for m in chat_history])

    chunks = "\n\n".join([c["chunk"] for c in chunks])
    chunks = chunks + "\n\n".join(["\n\n".join([c["chunk"] for c in msg["chunks"][:2]]) for msg in chat_history if msg["chunks"] != None])

    #chunks = translate_to_english(chunks)

    prompt=f"""
You are a specialist ASSISTANT in Movie titles and TV shows. Your task is to help USER find information from the SNIPPETS section and the CHAT conversation.
------------
SNIPPETS
{chunks}
------------
Your answer must be based solely on the SNIPPETS above and the CHAT history below.
Every part of the answer must be supported by the SNIPPETS only.
Your answer must be clear and detailed, bringing specific information from the SNIPPETS.
If you don't know the answer, just say that you don't know.
Don't make up an answer. If the answer is not within the SNIPPETS, say you don't know.
------------
CHAT:
{hist}
------------
Now write a JSON object with the following fields:
- "response":str, // the response for the chat with user
- "in_snippets":bool, //true if the answer is provided in the SNIPPETS. Otherwise, it must be always false.
- "in_chat":bool, // true if the answer is provided in the CHAT messages. Otherwise, it must always be false.
- "relevant_substrings": list[list[str,str]], // list of tuples, where each tuple must contain the direct quotes of relevant substrings from SNIPPETS, and the respective IDENTIFIER related to it.
Remember: Always provide the answer as a JSON object. Never reply as non-formatted text.
"""
    print("PROMPT IS",prompt)
    return prompt


def llm_api_(question):
    url = 'https://gcp-bindincapi-h2ppf7r6xa-ue.a.run.app/similarity'
    data = {
        "query": question
    }
    response = requests.post(url, json=data)
    return response.json()


# temp function
def llm_api(question):
    from langchain.vectorstores.faiss import FAISS
    from langchain.embeddings.vertexai import VertexAIEmbeddings

    embeddings = VertexAIEmbeddings(model_name="textembedding-gecko-multilingual@latest",chunk_size=1)

    store = FAISS.load_local("../data/index",embeddings,"index")

    result = store.similarity_search(question)

    return [{"chunk":r.page_content, "metadata":r.metadata} for r in result]
    
    

def reply(history):
    chunks = llm_api(history[-1]["text"])
    response = llm(build_prompt(history, chunks))
    response = "".join([row for row in response.split("\n") if "`" not in row])
    response = json.loads(response)
    print("RESPONSE IS",response)

    if not response["in_snippets"] and not response["in_chat"]:
        response["response"] = "Sorry, I was not provided with this information yet."
    return response["response"], chunks

