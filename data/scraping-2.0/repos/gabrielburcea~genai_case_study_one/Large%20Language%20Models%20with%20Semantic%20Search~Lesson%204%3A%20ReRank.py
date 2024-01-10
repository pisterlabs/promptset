# Databricks notebook source
"""Setup
Load needed API keys and relevant Python libaries."""

# COMMAND ----------



# COMMAND ----------

# !pip install cohere 
# !pip install weaviate-client

# COMMAND ----------

import os
from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv()) # read local .env file

# COMMAND ----------

import cohere
co = cohere.Client(os.environ['COHERE_API_KEY'])

# COMMAND ----------

import weaviate
auth_config = weaviate.auth.AuthApiKey(
    api_key=os.environ['WEAVIATE_API_KEY'])

# COMMAND ----------

client = weaviate.Client(
    url=os.environ['WEAVIATE_API_URL'],
    auth_client_secret=auth_config,
    additional_headers={
        "X-Cohere-Api-Key": os.environ['COHERE_API_KEY'],
    }
)

# COMMAND ----------

"""Dense Retrieval"""

# COMMAND ----------

from utils import dense_retrieval

# COMMAND ----------

query = "What is the capital of Canada?"

# COMMAND ----------

dense_retrieval_results = dense_retrieval(query, client)

# COMMAND ----------

from utils import print_result

# COMMAND ----------

print_result(dense_retrieval_results)

# COMMAND ----------

"""Improving Keyword Search with ReRank"""

# COMMAND ----------

from utils import keyword_search

# COMMAND ----------

query_1 = "What is the capital of Canada?"

# COMMAND ----------

query_1 = "What is the capital of Canada?"
results = keyword_search(query_1,
                         client,
                         properties=["text", "title", "url", "views", "lang", "_additional {distance}"],
                         num_results=3
                        )

for i, result in enumerate(results):
    print(f"i:{i}")
    print(result.get('title'))
    print(result.get('text'))

# COMMAND ----------

query_1 = "What is the capital of Canada?"
results = keyword_search(query_1,
                         client,
                         properties=["text", "title", "url", "views", "lang", "_additional {distance}"],
                         num_results=500
                        )

for i, result in enumerate(results):
    print(f"i:{i}")
    print(result.get('title'))
    #print(result.get('text'))

# COMMAND ----------

def rerank_responses(query, responses, num_responses=10):
    reranked_responses = co.rerank(
        model = 'rerank-english-v2.0',
        query = query,
        documents = responses,
        top_n = num_responses,
        )
    return reranked_responses

# COMMAND ----------

texts = [result.get('text') for result in results]
reranked_text = rerank_responses(query_1, texts)

# COMMAND ----------

for i, rerank_result in enumerate(reranked_text):
    print(f"i:{i}")
    print(f"{rerank_result}")
    print()

# COMMAND ----------

"""Improving Dense Retrieval with ReRank"""

# COMMAND ----------

from utils import dense_retrieval

# COMMAND ----------

query_2 = "Who is the tallest person in history?"

# COMMAND ----------

results = dense_retrieval(query_2,client)

# COMMAND ----------

for i, result in enumerate(results):
    print(f"i:{i}")
    print(result.get('title'))
    print(result.get('text'))
    print()

# COMMAND ----------

texts = [result.get('text') for result in results]
reranked_text = rerank_responses(query_2, texts)

# COMMAND ----------

for i, rerank_result in enumerate(reranked_text):
    print(f"i:{i}")
    print(f"{rerank_result}")
    print()

# COMMAND ----------

"""

Welcome to Lesson 4. I am very excited to show you one 
of my favorite methods called Rerank. Now, that you've 
learned keyword search and dense retrieval, Rerank 
is a way to improve them both, and it is actually the second component 
of semantic search in addition to 
dense retrieval. Rerank is a way for a large language model 
to sort search results from best to worst 
based on the relevance they have with respect 
to the query. 
Now, let's see Rerank in action in this lab. So, 
let's start by getting the API key from Cohere and Wediate. Then, we're 
going to import Cohere, and import Wediate. Next step, we are going 
to create our client that's going to store all the Wikipedia entries. 
Now, let's perform a search using DenseRetrieval, what you learned in the 
previous lesson. We're going to import the DenseRetrieval function. 
 
Now, let's search for the following query. What 
is the capital of Canada? We're going to call the answer DenseRetrievalResults, and 
to get these results we're going to 
use the DenseRetrieval function using the query, and the client 
as inputs. Now, if you remember we had a function that 
would help us print the results nicely. It was called 
print result. So, when we print the result, we get the 
following. 
So, let's take a look at these results. The 
first one is correct, It's Ottawa. Then, we have 
some noisy ones. Toronto is not the capital of Canada. And then, 
we also have Quebec city, which is the wrong answer. And why 
does this happen? Well, let me show you. Here's a small 
pictorial example. The answers are different, but it's for getting the 
idea. So, the query says, what is the capital 
of Canada? And let's say that the possible responses are these five. 
The capital of Canada is Ottawa, which is correct. Toronto is 
in Canada, which is correct, but irrelevant 
to the question. The capital of France is 
Paris, which is also correct, but not the answer to 
the question. Then, a wrong sentence. The capital of Canada is 
Sydney, which is not correct. And then, a sentence says the capital 
of Ontario is Toronto, which is true, but also 
not answering the questions. 
 
What happens when we do dense retrieval here? Well, let's 
say that these sentences are here. The capital of Canada is Ottawa, Toronto 
is in Canada, the capital of France is Paris, the 
capital of Canada is Sydney, and the capital of Ontario is Toronto. So, 
in an embedding, let's just imagine that they're located over 
here. Now, remember that the way dense retrieval 
works is it puts the query inside the embedding and then it 
returns the closest of the responses which in this case is the 
capital of Ontario is Toronto. 
Dense retrieval looks at similarities, so it returns the 
response that is the most similar to the question. 
This may not be the correct answer, this may not even be a true statement, 
it's just a sentence that happens to be close 
to the question semantically. So, therefore, dense retrieval has the 
potential to return things that are not necessarily the answer. How 
do we fix this? Well, this is where Rerank comes into play. 
 
Now, let me show you a small example of Rerank. Let's 
say that the query is what is the capital of Canada, and 
we have 10 possible answers, and as you can see some of these are 
relevant to the question and some of them are not. So, 
when we use dense retrieval, it gives us the top five, let's say, the five 
responses that are the most similar to the query. And let's 
say they are these ones over here, but we don't 
know which one is the response. 
We just have five sentences that are pretty 
close to the query. So, here is where Rerank comes into play. 
Rerank assigns to each query response pair a relevant 
score that tells you how relevant the answer is with respect to 
the query. It could also be a document. So, 
how relevant the document is with respect to the query. 
As you can see, the highest relevance here was 0.9, which 
corresponds to the capital of Canada is Ottawa, which 
is the correct answer. So, that is what Rerank does. 
You may be wondering how Rerank gets trained. Well, 
the way train re-rank is by giving it a 
lot of good pairs. So, that's a pair where the query and the 
response are very relevant, or when the query and document 
are very relevant, and training it to give those 
high relevance scores and then also giving it 
a big set of wrong query responses. So, query responses where the response 
does not correspond to the query. It may be close, but it 
may not correspond to it or also a document, a document 
that may not correspond to the query and 
if you train a model to give high scores to the good 
query response pairs and low scores to the 
bad query response pairs then you have the re-rank model that 
assigns a relevance and the relevance is high when you have a 
query and a response that are very related. 
Now, let's look at more re-rank examples. Let's use it 
to improve keyword search. So, we're going to import that keyword search 
function that we used in the first lesson. And 
again, let's ask it, what is the capital of 
Canada? So now, let's use keyword search to find 
the answers to this query. We're going to start by outputting three 
answers, and they're not so great. Monarchy of 
Canada, early modern period and flag of Canada. Why 
does it not work so well? Well, because keyword search is finding documents that have 
a lot of words in common with the 
query, but Keyword Search can't really tell if you're answering the questions. All 
these articles here have a lot of words in 
common with the query, but they're not the answer. 
 
So, let's make this a little bigger. Let's actually 
ask it for 500 results. And I'm not going 
to print the text, only the title. So here, we have 500 
top results. That's a lot of them. How do we find if one of these has 
the answer. Well, that's where re-rank comes into play. This function 
over here re-ranks the responses and it outputs the 
top 10. 
Now, let's call the answers texts and let's apply 
re-rank on texts where we have to input the query and the results. And finally, 
let's print the top 10 re-ranked results. Notice that this 
actually picked up the answer. It 
picked up Ottawa as the capital of Canada. 
The relevant score is very high. It's very close to one, It's 
0.98. Notice that the second best article is also pretty good because it 
talks about the different capitals that Canada has 
had in its history, and this one has 
a relevant score of 0.97. And as you can see, the third one 
is also pretty good, and Reranked actually picked the top 10 answers among 
the ones that keyword search surfaced that has the 
highest relevance, and now let's do 
a final example using dense retrieval. 
 
So again, I'm gonna use this dense retrieval function, and 
let's give it a slightly hard question. Let's give 
it the question who is the tallest person in history? This would be 
a hard question for keyword search because it may surface articles with 
the word history or the word person, it may not 
actually pick up the meaning of the question, but let's hope 
that dense retrieval can do better. So, we're going to call the 
function to give us some results. And now let's print out 
these results. Notice that this actually caught the 
correct answer. The person in history is Robert 
Wadlow, and that also picked up other documents, but we can still use 
re-rank to help us out. What happens when we re-rank these results? 
So, and now, let's call the re-rank function that again is 
going to check how relevant are the texts with 
respect to the query we gave it. And 
when we print the answers, then we get that 
indeed the one with the highest relevance 0.97 is the 
one belonging to Robert Guadalo. And for the 
other articles, it gave them some relevance, but 
it's not as high. 
 
So, Rerank actually helped us identify what is the correct 
answer to the question among the ones that dense 
retrieval had surfaced. Now, I encourage you to pause here, 
and actually try your own example. So, make your own queries, 
find the search results, and then, use Rerank to find the correct answers. 
Now, that we have all these search systems you may be wondering how 
to evaluate them. There are several ways to evaluate them, 
and some of them are mean average. Precision or MAP, 
Mean Reciprocal Rank or MRR, and Normalized Discounted 
Cumulative Gain or NDCG. 
Now, how would you make a test set for evaluating these 
models? Well, a good test set would be one containing queries, 
and correct responses, and then, you can compare 
these correct responses with the responses that the model 
gives you in a very similar way as you would find the 
accuracy or precision, or recall of a classification model. 
If you'd like to learn more about evaluating search systems, 
we are going to put some links to articles 
in the resources for you to take a look at more carefully. 
 
Now, that you've learned to use search, and re-rank to 
retrieve documents that contain the answer to a 
specific question, in the next lesson, you're going to 
learn something really cool. You're going to learn how to combine a 
search system a generation model in order to 
output the answer to a query in sentence 
mode the way a human would answer it. 


"""

# COMMAND ----------



# COMMAND ----------



# COMMAND ----------


