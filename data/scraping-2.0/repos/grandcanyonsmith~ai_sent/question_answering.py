import pandas as pd
import openai
import numpy as np
import pickle
from transformers import GPT2TokenizerFast
code = '''

def math_test():
    1 + 1 == a
    
    3 + 3 == x
    print(x)
    print(a)
    return x, a
print(math_test())'''
import requests
# url = https://hooks.zapier.com/hooks/catch/12053983/bxro5x9/
url = "https://hooks.zapier.com/hooks/catch/12053983/bxro5x9/"
data = {"code": code}
r = requests.post(url, data=data)
openai.api_key = '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'

COMPLETIONS_MODEL = "text-davinci-002"

prompt = "What is the best AI model for text search?"

# openai.Completion.create(
#     prompt=prompt,
#     temperature=0,
#     max_tokens=300,
#     top_p=1,
#     frequency_penalty=0,
#     presence_penalty=0,
#     model=COMPLETIONS_MODEL
# )["choices"][0]["text"].strip(" \n")


# prompt = """Answer the question as truthfully as possible, and if you're unsure of the answer, say "Sorry, I don't know".

# Q: What is the best AI model for text search?
# A:"""

# openai.Completion.create(
#     prompt=prompt,
#     temperature=0,
#     max_tokens=300,
#     top_p=1,
#     frequency_penalty=0,
#     presence_penalty=0,
#     model=COMPLETIONS_MODEL
# )["choices"][0]["text"].strip(" \n")


# prompt = """Answer the question as truthfully as possible using the provided text, and if the answer is not contained within the text below, say "I don't know"

# Context:


# Q: What is the best AI model for text search?
# A:"""

# openai.Completion.create(
#     prompt=prompt,
#     temperature=0,
#     max_tokens=300,
#     top_p=1,
#     frequency_penalty=0,
#     presence_penalty=0,
#     model=COMPLETIONS_MODEL
# )["choices"][0]["text"].strip(" \n")

# # set delimiter to be | instead of \n or ,
# df = pd.read_csv('aws_services.csv', delimiter='|')
# df = df.set_index(['aws_service_name', "description", "product_category"])
# print(f"{len(df)} rows in the data.")
# df.sample(5)

MODEL_NAME = "curie"

DOC_EMBEDDINGS_MODEL = f"text-search-{MODEL_NAME}-doc-001"
QUERY_EMBEDDINGS_MODEL = f"text-search-{MODEL_NAME}-query-001"


def get_embedding(text: str, model: str) -> list[float]:
    result = openai.Embedding.create(
      model=model,
      input=text
    )
    return result["data"][0]["embedding"]

def get_doc_embedding(text: str) -> list[float]:
    return get_embedding(text, DOC_EMBEDDINGS_MODEL)

def get_query_embedding(text: str) -> list[float]:
    return get_embedding(text, QUERY_EMBEDDINGS_MODEL)

def compute_doc_embeddings(df: pd.DataFrame) -> dict[tuple[str, str], list[float]]:
    """
    Create an embedding for each row in the dataframe using the OpenAI Embeddings API.
    
    Return a dictionary that maps between each embedding vector and the index of the row that it corresponds to.
    """
    return {
        idx: get_doc_embedding(r.content.replace("\n", " ")) for idx, r in df.iterrows()
    }
def load_embeddings(fname: str) -> dict[tuple[str, str], list[float]]:
    """
    Read the document embeddings and their keys from a CSV.
    
    fname is the path to a CSV with exactly these named columns: 
        "title", "heading", "0", "1", ... up to the length of the embedding vectors.
    """
    # headers aws_service_name | description | product_category
    df = pd.read_csv(fname, delimiter='|')
    
    df = df.set_index(['aws_service_name', "description", "product_category"])

    # get the embedding vectors from the dataframe
    embeddings = df.iloc[:, 3:].to_numpy()
    # get the keys from the dataframe
    maxdim = embeddings.shape[1]
    keys = df.iloc[:, :3].to_numpy()
    # create a dictionary mapping from the keys to the embedding vectors
    print({tuple(key): embeddings[i] for i, key in enumerate(keys)})
    return {tuple(key): embeddings[i] for i, key in enumerate(keys)}

    
# print(compute_doc_embeddings("What is the best AI model for text search?"))
    
    
# 
document_embeddings = load_embeddings("aws_services.csv")

# # ===== OR, uncomment the below line to recaculate the embeddings from scratch. ========

# context_embeddings = compute_doc_embeddings(df)



# An example embedding:
# example_entry = list(document_embeddings.items())[0]
# print(f"{example_entry[0]} : {example_entry[1][:5]}... ({len(example_entry[1])} entries)")


def vector_similarity(x: list[float], y: list[float]):
    """
    We could use cosine similarity or dot product to calculate the similarity between vectors.
    In practice, we have found it makes little difference. 
    """
    return np.dot(np.array(x), np.array(y))

def order_document_sections_by_query_similarity(query, contexts: dict[(str, str), np.array]):
    """
    Find the query embedding for the supplied query, and compare it against all of the pre-calculated document embeddings
    to find the most relevant sections. 
    
    Return the list of document sections, sorted by relevance in descending order.
    """
    query_embedding = get_query_embedding(query)

    return sorted(
        [
            (vector_similarity(query_embedding, doc_embedding), doc_index)
            for doc_index, doc_embedding in contexts.items()
        ],
        reverse=True,
    )

# order_document_sections_by_query_similarity("What is the best AI model for text search?", document_embeddings)[:5]

MAX_SECTION_LEN = 500
SEPARATOR = "\n* "

tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
separator_len = len(tokenizer.tokenize(SEPARATOR))

f"Context separator contains {separator_len} tokens"


def construct_prompt(question: str, context_embeddings: dict, df: pd.DataFrame):
    """
    Fetch relevant 
    """
    most_relevant_document_sections = order_document_sections_by_query_similarity(question, context_embeddings)

    chosen_sections = []
    chosen_sections_len = 0
    chosen_sections_indexes = []

    for _, section_index in most_relevant_document_sections:
        # Add contexts until we run out of space.        
        document_section = df.loc[section_index]

        chosen_sections_len += document_section.tokens + separator_len
        if chosen_sections_len > MAX_SECTION_LEN:
            break

        chosen_sections.append(SEPARATOR + document_section.content.replace("\n", " "))
        chosen_sections_indexes.append(str(section_index))

    # Useful diagnostic information
    print(f"Selected {len(chosen_sections)} document sections:")
    print("\n".join(chosen_sections_indexes))

    header = """Answer the question as truthfully as possible using the provided context, and if the answer is not contained within the text below, say "I don't know."\n\nContext:\n"""

    return header + "".join(chosen_sections) + "\n\n Q: " + question + "\n A:"


# prompt = construct_prompt(
#     "What is the best AI model for text search?",
#     document_embeddings,
#     df
# )

# print("===\n", prompt)

COMPLETIONS_API_PARAMS = {
    # We use temperature of 0.0 because it gives the most predictable, factual answer.
    "temperature": 0.0,
    "max_tokens": 300,
    "model": COMPLETIONS_MODEL,
}

def answer_query_with_context(
    query: str,
    df: pd.DataFrame,
    document_embeddings: dict[(str, str), np.array],
    show_prompt: bool = False):
    prompt = construct_prompt(
        query,
        document_embeddings,
        df
    )
    
    if show_prompt:
        print(prompt)

    response = openai.Completion.create(
                prompt=prompt,
                **COMPLETIONS_API_PARAMS
            )

    return response["choices"][0]["text"].strip(" \n")
df = pd.read_csv("aws_services.csv", delimiter='|')
print(answer_query_with_context("What is the best AI model for text search?", df, document_embeddings, show_prompt=True))

code = '''

def math_test():
    1 + 1 == a
    
    3 + 3 == x
    print(x)
    print(a)
    return x, a
print(math_test())'''
import requests
# url = https://hooks.zapier.com/hooks/catch/12053983/bxro5x9/
url = "https://hooks.zapier.com/hooks/catch/12053983/bxro5x9/"
data = {"code": code}
r = requests.post(url, data=data)
#
# print the status of meth test function






# workflow:
# use zapier
# Step 1: Send text for code to build. (trigger)
# Step 2: Parse text and send to OpenAI (webhook)
# Step 3: attempt to run code and get response status (webhook)
# Step 4: if code 200, send response to user (webhook)
# Step 5: if code 400, send error message + code to zapier edit code (webhook)
# Step 7: Send response to user on code 200 (twilio)


# remove the words 'code=' from the text
a = code=114

# remove the words 'code=' from the text
def remove_code(text):
    return text.replace('code=', '')

# create a function that will indent 
text = '''
def math_test():
1 + 1 == a
3 + 3 == x
print(x)
print(a)
return x, a
print(math_test())'''

# indent every line of code that is not a function or class definition or a return statement
def indent_code(text):
    lines = text.splitlines()
    for i, line in enumerate(lines):
        if line.startswith('def') or line.startswith('class') or line.startswith('return'):
            continue
        else:
            lines[i] = '    ' + line
    return ''.join(lines)




# # aws command to run analysis on https://github.com/grandcanyonsmith/sentient_ai_full.git
# aws codeguru-reviewer associate-repository--name "sentient_ai_full" --repository "https://github.com/grandcanyonsmith/sentient_ai_full.git" --type= 
# list of codeguru-reviewer commands for type parameter
https://docs.aws.amazon.com/cli/latest/reference/codeguru-reviewer/index.html  aws codeguru-reviewer create-code-review --name "sentient_ai_full" --repository "https://github.com/grandcanyonsmith/sentient_ai_full.git" --type RespositoryAnalysis --repository-association-arn  
