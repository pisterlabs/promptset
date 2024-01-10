import pandas
import tiktoken
import openai
import concurrent.futures
from openai.embeddings_utils import distances_from_embeddings

openai.api_key = "**"


with open('data/news.json') as news:
    df = pandas.read_json(news, engine='ujson', encoding='ISO-8859-1')

df.describe()
tokenizer = tiktoken.get_encoding("cl100k_base")
df['n_tokens'] = df.text.apply(lambda x: len(tokenizer.encode(x)))
df.head()
print(sum(df['n_tokens']))


def create_embedding(text):
    return openai.Embedding.create(input=text, engine='text-embedding-ada-002')['data'][0]['embedding']


#with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
#    results = list(executor.map(create_embedding, df.text))
#df['embeddings'] = results
#df.to_json('data/embeddings.json')

with open('data/embeddings.json') as news:
    df = pandas.read_json(news, engine='ujson', encoding='ISO-8859-1')


def create_context(
        question, df, max_len=1800, size="ada"
):
    """
    Create a context for a question by finding the most similar context from the dataframe
    """
    # Get the embeddings for the question
    q_embeddings = openai.Embedding.create(input=question, engine='text-embedding-ada-002')['data'][0]['embedding']

    # Get the distances from the embeddings
    df['distances'] = distances_from_embeddings(q_embeddings, df['embeddings'].values, distance_metric='cosine')

    returns = []
    cur_len = 0

    # Sort by distance and add the text to the context until the context is too long
    for i, row in df.sort_values('distances', ascending=True).iterrows():

        # Add the length of the text to the current length
        cur_len += row['n_tokens'] + 4

        # If the context is too long, break
        if cur_len > max_len:
            break

        # Else add it to the text that is being returned
        returns.append(row["text"])

    # Return the context
    return "\n\n###\n\n".join(returns)


def answer_question(
        df,
        model="text-davinci-003",
        question="show me a random magic the gathering card?",
        max_len=10000,
        size="ada",
        max_tokens=200,
        stop_sequence=None
):
    """
    Answer a question based on the most similar context from the dataframe texts
    """
    context = create_context(question, df, max_len=max_len, size=size)

    try:
        # Create a completions using the question and context
        response = openai.Completion.create(
            prompt=f"Answer the question based on the context below, and if the question can't be answered based on the context then don't use the context.\"\n\nContext: {context}\n\n---\n\nQuestion: {question}\nAnswer:",
            temperature=0.4,
            max_tokens=max_tokens,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
            stop=stop_sequence,
            model=model,
        )
        return response["choices"][0]["text"].strip()
    except Exception as e:
        print(e)
        return ""


while True:
    question = input("Ask me a question: ")
    print(answer_question(df, question=question))



