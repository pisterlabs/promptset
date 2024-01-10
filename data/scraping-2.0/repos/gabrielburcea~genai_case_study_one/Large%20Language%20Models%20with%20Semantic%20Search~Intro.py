# Databricks notebook source

"""Welcome to this short course, Large Language Models 
with Semantic Search, built in partnership with Cohere. 
In this course, you'll learn how to incorporate large language models, or 
LLMs, into information search in your own applications. 
For example, let's say you run a website with a lot of articles, picture 
Wikipedia for the sake of argument, or 
a website with a lot of e-commerce products. 
Even before LLMs, it was common to have keyword search to 
let people maybe search your site. 
But with LLMs, you can now do much more. 
First, you can let users ask questions that your 
system then searches your site or database to answer. 
Second, the LLM is also making the retrieve results more 
relevant to the meaning or the semantics of what the user is 
asking about. 
I'd like to introduce the instructors for this course, Jay 
Allamar and Luis Serrano. 
Both Jay and Luis are experienced machine learning engineers 
as well as educators. 
I've admired for a long time some highly referenced illustrations 
that Jay had created to explain 
transformer networks. 
He's also co-authoring a book, Hands-On Large Language Models. 
Luis is the author of the book, Grokking Machine Learning, 
and he also taught with DeepLearning.ai. 
Math for Machine Learning. 
At Cohere, Jay and Luis, together with Neil Amir, 
have also been working on a site called. 
LLMU, and have a lot of experience teaching developers to use LLMs. 
So I was thrilled when they agreed to 
teach semantic search with LLMs. 
Thanks, Andrew. 
What an incredible honor it is to be 
teaching this course with you. 
Your machine learning course introduced me to machine 
learning eight years ago, and continues to be 
an inspiration to continue sharing what I learn. 
As you mentioned, Luis and I work at Cohere, 
so we get to advise others in the industry on how to 
use and deploy large language models for various 
use cases. 
We are thrilled to be doing this course 
to give developers the tools they need to 
build robust LLM powered apps. 
We're excited to share what we learned from our 
experience in the field. 
Thank you, Jay and Luis. 
Great to have you with us. 
This course consists of the following topics. 
First, it shows you how to use basic keyword search, 
which is also called lexical search, which powered a lot of search 
systems before large language models. 
It consists of finding the documents that has 
the highest amount of matching words with the query. 
Then you learn how to enhance this type 
of keyword search with a method called re-rank. 
As the name suggests, this then ranks the responses 
by relevance with the query. 
After this, you learn a more advanced method of search, 
which has vastly improved the results of keyword search, 
as it tries to use the actual meaning 
or the actual semantic meaning of the text 
with which to carry out the search. 
This method is called dense retrieval. 
This uses a very powerful tool in natural 
language processing called embeddings, which is a way to 
associate a vector of numbers with every piece of text. 
Semantic search consists of finding the closest documents 
to the query in the space of embeddings. 
Similar to other models, search algorithms need 
to be properly evaluated. 
You also learn effective ways to do this. 
Finally, since LLMs can be used to generate answers, 
you also learn how to plug in the search results into an 
LLM and have it generate an answer based on them. 
Dense retrieval with embeddings vastly improves 
the question answering capabilities of an LLM as it 
first searches for and retrieves the 
relevant documents and it creates an answer from this 
retrieved information. 
Many people have contributed to this course. 
We're grateful for the hard work of Meor Amer, Patrick 
Lewis, Nils Reimer, and Sebastian Hofstatter from Cohere, as 
well as on the DeepLearning.ai side, Eddie Shyu 
and Diala Ezzedine. 
In the first lesson, you will see how search was done before large 
language models. 
From there, we will show you how to improve search using LLMs, 
including tools such as embeddings and re-rank. 
That sounds great. 
And with that, let's dive in and go on to the next video. """
