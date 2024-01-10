# Databricks notebook source
# MAGIC %md
# MAGIC
# MAGIC # Embeddings, Vector Databases, and Search
# MAGIC
# MAGIC Converting text into embedding vectors is the first step to any text processing pipeline. As the amount of text gets larger, there is often a need to save these embedding vectors into a dedicated vector index or library, so that developers won't have to recompute the embeddings and the retrieval process is faster. We can then search for documents based on our intended query and pass these relevant documents into a language model (LM) as additional context. We also refer to this context as supplying the LM with "state" or "memory". The LM then generates a response based on the additional context it receives! 
# MAGIC
# MAGIC In this notebook, we will implement the full workflow of text vectorization, vector search, and question answering workflow. While we use [FAISS](https://faiss.ai/) (vector library) and [ChromaDB](https://docs.trychroma.com/) (vector database), and a Hugging Face model, know that you can easily swap these tools out for your preferred tools or models!
# MAGIC
# MAGIC <img src="https://files.training.databricks.com/images/llm/updated_vector_search.png" width=1000 target="_blank" > 
# MAGIC
# MAGIC ### ![Dolly](https://files.training.databricks.com/images/llm/dolly_small.png) Learning Objectives
# MAGIC 1. Implement the workflow of reading text, converting text to embeddings, saving them to FAISS and ChromaDB 
# MAGIC 2. Query for similar documents using FAISS and ChromaDB 
# MAGIC 3. Apply a Hugging Face language model for question answering!

# COMMAND ----------

# MAGIC %run ./resources/helper

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 1: Reading data
# MAGIC
# MAGIC In this section, we are going to use 2022 annual report from Bank of America.
# MAGIC The report is stored on Unity Catalog's managed volume. </a>.

# COMMAND ----------



# COMMAND ----------

# Extract text
from PyPDF2 import PdfReader

file_path2 = '/Volumes/cjc_cap_markets/capm_data/10kreports/boa-2022-10k.pdf'

reader = PdfReader(file_path2)
page = reader.pages[0]

print(page.extract_text()[:400])

# COMMAND ----------

# MAGIC %md
# MAGIC ## Vector Library: FAISS
# MAGIC
# MAGIC Vector libraries are often sufficient for small, static data. Since it's not a full-fledged database solution, it doesn't have the CRUD (Create, Read, Update, Delete) support. Once the index has been built, if there are more vectors that need to be added/removed/edited, the index has to be rebuilt from scratch. 
# MAGIC
# MAGIC That said, vector libraries are easy, lightweight, and fast to use. Examples of vector libraries are [FAISS](https://faiss.ai/), [ScaNN](https://github.com/google-research/google-research/tree/master/scann), [ANNOY](https://github.com/spotify/annoy), and [HNSM](https://arxiv.org/abs/1603.09320).
# MAGIC
# MAGIC FAISS has several ways for similarity search: L2 (Euclidean distance), cosine similarity. You can read more about their implementation on their [GitHub](https://github.com/facebookresearch/faiss/wiki/Getting-started#searching) page or [blog post](https://engineering.fb.com/2017/03/29/data-infrastructure/faiss-a-library-for-efficient-similarity-search/). They also published their own [best practice guide here](https://github.com/facebookresearch/faiss/wiki/Guidelines-to-choose-an-index).
# MAGIC
# MAGIC If you'd like to read up more on the comparisons between vector libraries and databases, [here is a good blog post](https://weaviate.io/blog/vector-library-vs-vector-database#feature-comparison---library-versus-database).

# COMMAND ----------

# MAGIC %md
# MAGIC The overall workflow of FAISS is captured in the diagram below. 
# MAGIC
# MAGIC <img src="https://miro.medium.com/v2/resize:fit:1400/0*ouf0eyQskPeGWIGm" width=700>
# MAGIC
# MAGIC Source: [How to use FAISS to build your first similarity search by Asna Shafiq](https://medium.com/loopio-tech/how-to-use-faiss-to-build-your-first-similarity-search-bf0f708aa772).

# COMMAND ----------

# MAGIC %md
# MAGIC ### Load open source text vectorization library. 
# MAGIC Alternatively, OpenAI has a popular embeddings model - for most of 2023 the ada002v2 model is their defacto recommendation

# COMMAND ----------

from sentence_transformers import InputExample

faiss_train_examples2 = []
all_text = []

for page in reader.pages:
  # for each page in the pdf, create an embeddings vector, and seperate list fro the pages of text
  faiss_train_examples2.append(InputExample(texts=page.extract_text())) # create a list of embeddings
  all_text.append(page.extract_text())  # create a list of text

# COMMAND ----------

# Get word count
import pandas as pd

temp = pd.DataFrame(data=all_text, columns=['page_text'])
temp['count'] = temp.page_text.apply(lambda x: len(x.split()))
display(temp)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Chunking window optimization
# MAGIC Convert nested list of pages into nested list of chunks of a specified number of words

# COMMAND ----------

def combine_nested_list(nested_list):
  """helper function to combine list"""
  combined_text = ' '.join([''.join(sublist) for sublist in nested_list])
  return combined_text


def split_string_with_overlap(input_string, chunk_size=500, overlap=20):
  """helper function to (re)split into specific chunk sizes. Overlap refers to number of shared words across sliding windows to prevent context loss when chunking. chunk size should be mostly based on model context window size, and to a lesser extent, desired number of chunks returned per query and considerations around context window loss"""
  words = input_string.split()
  result = []
  start = 0

  while start < len(words):
    end = min(start + chunk_size, len(words))
    chunk = words[start:end]

    if start > 0:
        chunk = words[start-overlap:end]

    result.append(" ".join(chunk))
    start += chunk_size - overlap

  return result

# COMMAND ----------

# Re-window document

chunk_size=1000
overlap=30

tempa = combine_nested_list(all_text) # reduce nested list into single string
tempb = split_string_with_overlap(tempa, chunk_size=chunk_size, overlap=overlap) # apply sliding window for chunking
pdf_subset2 = pd.DataFrame(data=tempb, columns=['text']) # create a dataframe
pdf_subset2['count'] = pdf_subset2.text.apply(lambda x: len(x.split())) # create new column to count/confirm number of words per chunk
pdf_subset2["id"] = pdf_subset2.index # add an id column

display(pdf_subset2)

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ### Step 2: Vectorize text into embedding vectors
# MAGIC We will be using `Sentence-Transformers` [library](https://www.sbert.net/) to load a language model to vectorize our text into embeddings. The library hosts some of the most popular transformers on [Hugging Face Model Hub](https://huggingface.co/sentence-transformers).
# MAGIC Here, we are using the `model = SentenceTransformer("all-MiniLM-L6-v2")` to generate embeddings.

# COMMAND ----------

# MAGIC %md Make sure to run cell below for access to api

# COMMAND ----------

import openai
from openai.embeddings_utils import get_embedding, cosine_similarity 
import tiktoken

openaikey = dbutils.secrets.get("tokens", "canadaeh-openaikey")
openai.api_key = openaikey
openai.api_type = "azure"
openai.api_base = "https://canada-eh-openai.openai.azure.com/"
openai.api_version = "2023-07-01-preview"
aoai_model = "gpt-35-deployment"


# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ### Step 3: Saving embedding vectors to FAISS index
# MAGIC Below, we create the FAISS index object based on our embedding vectors, normalize vectors, and add these vectors to the FAISS index. 

# COMMAND ----------

pdf_subset2["text"]

# COMMAND ----------

def get_embedding_delay(x):
  time.sleep(1)
  return(get_embedding(x, engine = 'cjc-text-embedding-ada-002'))

# COMMAND ----------

import time

pdf_subset2['ada_v2'] = pdf_subset2["text"].apply(lambda x : get_embedding_delay(x)) 

# engine should be set to the deployment name you chose when you deployed the text-embedding-ada-002 (Version 2) model

#can take a minute or so to run based on the number of embeddings being generated (each a seperate api call)

# COMMAND ----------

spark_df = spark.createDataFrame(pdf_subset2)
spark_df.write.mode("overwrite").saveAsTable("cjc_cap_markets.capm_data.ada_embeddings")

# COMMAND ----------

# MAGIC %md ### Search through documents

# COMMAND ----------

# search through the reviews for a specific product
def search_docs(df, user_query, top_n=3):
  embedding = get_embedding(
      user_query,
      engine="cjc-text-embedding-ada-002" # engine should be set to the deployment name you chose when you deployed the text-embedding-ada-002 (Version 2) model
  )
  df["similarities"] = df.ada_v2.apply(lambda x: cosine_similarity(x, embedding))

  res = (
      df.sort_values("similarities", ascending=False)
      .head(top_n)
  )
  return res

# COMMAND ----------

res = search_docs(pdf_subset2, "Can I get information on cable company tax revenue?", top_n=4)
display(res['text'])

# COMMAND ----------

# MAGIC %md
# MAGIC ## Prompt engineering for question answering 
# MAGIC
# MAGIC Now that we have identified documents about space from the news dataset, we can pass these documents as additional context for a language model to generate a response based on them! 
# MAGIC
# MAGIC We first need to pick a `text-generation` model. Below, we use a Hugging Face model. You can also use OpenAI as well, but you will need to get an Open AI token and [pay based on the number of tokens](https://openai.com/pricing). 

# COMMAND ----------

#Set Model

# aoai_model = "gpt-35-turbo-16k" # gpt-35-turbo" # Specify the language model
#aoai_model = "oneenvgpt4" # GPT-4" # Specify the language model

# COMMAND ----------

prompts = ["who is the primary company within this 10-K Form?",
          "where is bank of america headquartered?",
          "what were bank of america's total assets and liabilities",
          "what were bank of america's total revenues",
          "what are bank of america's positions on diversity and inclusion?",
          "how did bank of america's net earnings improve year over year",
          "what are bank of america's top assets",
          "what were the q4 earnings of bank of america?"]

# COMMAND ----------

# MAGIC %run ./resources/prettify

# COMMAND ----------

def q_and_a(query, num_chunks):
  res = search_docs(pdf_subset2, query, top_n=num_chunks)['text'].tolist()
  context = combine_nested_list(res)

  prompt = [{"role": "system", "content": "You are a helpful assistant."},
            {'role': 'user','content':'test'}] 
  prompt[1] = {'role': 'user','content': f'Please answer the question: {query}. If the answer is not in the below context, feel free to attempt to answer based on your internal knowledge, but be clear that you are doing so. \n\n provided context: {context}' }

  result = openai.ChatCompletion.create(
      engine=aoai_model,
      messages=prompt,
      max_tokens=300,
      temperature=0.0,
  )

  display_answer(query, result)

# COMMAND ----------

for i in prompts:
  q_and_a(i, 4)

# COMMAND ----------


