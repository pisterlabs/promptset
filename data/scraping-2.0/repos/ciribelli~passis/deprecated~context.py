import numpy as np
import pandas as pd
import openai
#from openai.embeddings_utils import distances_from_embeddings
from scipy.spatial.distance import cosine
from dotenv import load_dotenv
import os

first_item = ""
def create_context(
    question, df, max_len=1200, size="ada"
):
    """
    Create a context for a question by finding the most similar context from the dataframe
    """
    load_dotenv()
    openai.api_key = os.getenv('OPENAI_API_KEY')

    # Get the embeddings for the question
    q_embeddings = openai.Embedding.create(input=question, engine='text-embedding-ada-002')['data'][0]['embedding']

    # Get the distances from the embeddings - deprecated pela OpeanAI
    # df['distances'] = distances_from_embeddings(q_embeddings, df['embeddings'].values, distance_metric='cosine')]
    # nova alternativa para distancia cossenoidal:
    df["distances"] = df["embeddings"].apply(lambda x: cosine(q_embeddings, x))

    returns = []
    links = []
    cur_len = 0

    # Sort by distance and add the text to the context until the context is too long
    for i, row in df.sort_values('distances', ascending=True).iterrows():

        # Add the length of the text to the current length
        cur_len += row['n_tokens'] + 4

        # If the context is too long, break
        if cur_len > max_len:
            break

        # Else add it to the text that is being returned
        returns.append(row["texto"])
        links.append(str(row["tabela"])+'/'+str(row["index"]))

    global first_item
    first_item = links[0]
    # Return the context
    return "\n🤖\n".join(returns)


def answer_question(
    df,
    model="gpt-3.5-turbo-instruct",
    question="Am I allowed to publish model outputs to Twitter, without a human review?",
    max_len=1200,
    size="ada",
    debug=True,
    max_tokens=240,
    #stop_sequence="\n"
):
    """
    Answer a question based on the most similar context from the dataframe texts
    """
    context = create_context(question, df, max_len=max_len, size=size,)

    if debug:
        print("Context:\n" + context)
        print("\n\n")

    try:
        # Create a completions using the question and context
        response = openai.Completion.create(
            prompt=f"Você é meu assistente virtual para assuntos pessoais e me ajuda com ideias e lembretes sobre minha rotina e o que acontece no mundo. Você receberá uma série de notas pessoais e informações a meu respeito abaixo, e deverá elaborar a resposta com base nesses dados. Se não souber a respota, pode buscar a melhor aproximação: \n\n Meus lembretes e informações: {context}\n\n---\n\nAgora, responda essa pergunta: {question}\n",
            temperature=0.3,
            max_tokens=max_tokens,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
            #stop="20",
            model=model,
        )
        return response["choices"][0]["text"].strip()
    except Exception as e:
        print(e)
        return ""

def responde_emb(pergunta, dados):
    df = pd.DataFrame(dados)
    print(df)
    df['embeddings'] = df['embeddings'].apply(np.array)
    resposta = answer_question(df, question=pergunta).replace("\n", '<br>')
    saida = resposta.replace("<br>", "\n")
    global first_item
    return saida, first_item

