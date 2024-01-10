import openai
import time
import numpy as np
import openai
import pandas as pd
import pickle
import tiktoken
import os
from quickchart import QuickChart
COMPLETIONS_MODEL = "text-davinci-003"
EMBEDDING_MODEL = "text-embedding-ada-002"
from dotenv import load_dotenv
load_dotenv()

API_KEY = os.getenv("OPENAI_API_KEY") 
openai.api_key = API_KEY
df = pd.read_csv('ICICI2.csv')

# df["token"] = None
# for idx, r in df.iterrows():
# #     print(len(r.content))
# #     df["token"] = df[len(r.content)]
#     df.loc[idx,'token'] = len(r.content)

# "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" -- MLAI
# "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" -- MLAI
# "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" -- LP
def get_embedding(text: str, model: str=EMBEDDING_MODEL) -> list[float]:
    # openai.api_key = "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"
    result = openai.Embedding.create(
      model=model,
      input=text
    )
    return result["data"][0]["embedding"]

# import time
# def compute_doc_embeddings(df: pd.DataFrame) -> dict[tuple[str, str], list[float]]:
#     embeddings = {}
#     for idx, r in df.iterrows():
#         print(idx)
# #         print(r)
#         embeddings[idx] = get_embedding(r.title)
#         time.sleep(5)  # Add a delay of 10 second between requests
#     return embeddings


def load_embeddings(fname: "str") -> dict[tuple[str, str], list[float]]:
    """
    Read the document embeddings and their keys from a CSV.

    fname is the path to a CSV with exactly these named columns:
        "title", "heading", "0", "1", ... up to the length of the embedding vectors.
    """

    df = pd.read_csv(fname, header=0)

    max_dim = max([int(c) for c in df.columns if c != "title"])
    return {
        (r.title): [r[str(i)] for i in range(max_dim + 1)] for _, r in df.iterrows()

    }

document_embeddings = load_embeddings("ICICI_embed_5.csv")


def vector_similarity(x: list[float], y: list[float]) -> float:
    """
    Returns the similarity between two vectors.

    Because OpenAI Embeddings are normalized to length 1, the cosine similarity is the same as the dot product.
    """
    return np.dot(np.array(x), np.array(y))


def order_document_sections_by_query_similarity(query: str, contexts: dict[(str, str), np.array]) -> list[
    (float, (str, str))]:
    """
    Find the query embedding for the supplied query, and compare it against all of the pre-calculated document embeddings
    to find the most relevant sections.

    Return the list of document sections, sorted by relevance in descending order.
    """
    query_embedding = get_embedding(query)

    document_similarities = sorted([
        (vector_similarity(query_embedding, doc_embedding), doc_index) for doc_index, doc_embedding in contexts.items()
    ], reverse=True)

    return document_similarities

# a = order_document_sections_by_query_similarity("Give me a list of icici credit crads  interest rate", document_embeddings)[:5]

MAX_SECTION_LEN = 7552
SEPARATOR = "\n* "
ENCODING = "gpt2"  # encoding for text-davinci-003

encoding = tiktoken.get_encoding(ENCODING)
separator_len = len(encoding.encode(SEPARATOR))

print(f"Context separator contains {separator_len} tokens")


def construct_prompt(question: str, context_embeddings: dict, df: pd.DataFrame) -> str:
    """
    Fetch relevant
    """
    most_relevant_document_sections = order_document_sections_by_query_similarity(question, context_embeddings)[:3]
    #     print(most_relevant_document_sections)
    chosen_sections = []
    chosen_sections_len = 0
    chosen_sections_indexes = []

    for _, section_index in most_relevant_document_sections:
        # Add contexts until we run out of space.
        #         print(section_index)
        document_section = df.loc[section_index]
        #         print(document_section)
        chosen_sections_len += document_section.token + separator_len
        if chosen_sections_len > MAX_SECTION_LEN:
            break

        chosen_sections.append(SEPARATOR + document_section.content.replace("\n", " "))
        chosen_sections_indexes.append(str(section_index))

    # Useful diagnostic information
    print(f"Selected {len(chosen_sections)} document sections:")
    print("\n".join(chosen_sections_indexes))
    # print(type("".join(chosen_sections) ))
    # header = """Answer the question as truthfully as possible using the provided Given Below Information and previous conversation, and if the answer is not contained within in both, say "I don't know.For more details call on our CUSTOMER CARE NO.1800 1080"\n\nGiven Below Information:"""
    header = "".join(chosen_sections) + """\nPlease find a relevent data from Above text and give me response in this formate only and only change labels and data using above data. \n
data: {{
    labels: ['Q1', 'Q2', 'Q3', 'Q4'],
    datasets: [{{
      label: 'Users',
      data: [50, 60, 70, 180]
    }}, {{
      label: 'Revenue',
      data: [100, 200, 300, 400]
    }}]
  }}
Important Note: you must give response for all given data.Data must in Numeric or Flot no extra symbol or Albhabets
make changes according to user Question : {question}""".format(question=question)

    return header 

COMPLETIONS_API_PARAMS = {
    # We use temperature of 0.0 because it gives the most predictable, factual answer.
    "temperature": 1.0,
    "max_tokens": 500,
    "model": COMPLETIONS_MODEL,
}

def answer_query_with_context(
        query: str,
        df: pd.DataFrame,
        document_embeddings: document_embeddings,
        show_prompt: bool = True,
) -> str:
    # Construct prompt to send to OpenAI API
    prompt = construct_prompt(
        query,
        document_embeddings,
        df
    )

    # Print prompt if show_prompt is True
    if show_prompt:
        print(prompt)

    # Send prompt to OpenAI API and get response
    response = openai.Completion.create(
        prompt=prompt,
        **COMPLETIONS_API_PARAMS
    )
    text = response["choices"][0]["text"]
    print(text)
    return text

def chart1(text):
    # print(resp)
    resp = answer_query_with_context(text, df, document_embeddings)
    qc = QuickChart()
    qc.width = 600
    qc.height = 300
    qc.device_pixel_ratio = 2.0
    qc.config = '''{{
    type: 'bar',
    {resp},
    options: {{
        title: {{
        display: true,
        text: 'Bar Chart',
        }},
        plugins: {{
        datalabels: {{
            anchor: 'center',
            align: 'center',
            color: '#666',
            font: {{
            weight: 'normal',
            }},
        }},
        }},
    }}   
    }}'''.format(resp=resp)
    # print(qc.config)
    timestamp = str(int(time.time()))  # Get the current timestamp as a string

    image_filename = 'static\images\mychart_' + timestamp + '.png'  # Construct the unique image file name

    url = qc.get_short_url()
    print(qc.get_short_url())

    image = qc.to_file(image_filename) 
    # url = qc.get_short_url()
    # print(qc.get_short_url())
    # image = qc.to_file('A:\icici_gpt\static\images\mychart2.png')
    
    return url , image_filename
    
    