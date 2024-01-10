from langchain import OpenAI, PromptTemplate
from langchain.docstore.document import Document
from langchain.prompts import PromptTemplate
from langchain.chains.summarize import load_summarize_chain
import tiktoken
import os
from langchain.chains.qa_with_sources import load_qa_with_sources_chain
import pprint

pp = pprint.PrettyPrinter(indent=4)


llm = OpenAI(model_name="text-davinci-003", temperature=0, max_tokens=, batch_size=1)

enc = tiktoken.get_encoding("cl100k_base")

files = [f"{n}.txt" for n in range(1, 71)]

# Get the summary files
files = [file for file in os.listdir("summaries") if file.endswith(".txt")]

docs = [Document(page_content=open(f"summaries/{file}").read()) for file in files]

chain = load_qa_with_sources_chain(llm, chain_type="map_reduce")


# question_prompt_template = """Use the following summaries of news articles to create a professional summarized news report for a business executive. It should group relevant information together and be written in a concise, professional manner. Provide a bullet point list of each of the different categories. Come up with relevant categories based on the summaries.
# {context}
# News Articles: {question}
# """
# QUESTION_PROMPT = PromptTemplate(template=question_prompt_template, input_variables=["context", "question"])

# combine_prompt_template = """Given the following extracted parts of a long document and a question, create a final answer with references ("SOURCES").
# If you don't know the answer, just say that you don't know. Don't try to make up an answer.
# ALWAYS return a "SOURCES" part in your answer.
# Respond in Italian.

# QUESTION: {question}
# =========
# {summaries}
# =========
# FINAL ANSWER IN ITALIAN:"""
# COMBINE_PROMPT = PromptTemplate(template=combine_prompt_template, input_variables=["summaries", "question"])

# chain = load_qa_with_sources_chain(
#     llm,
#     chain_type="map_reduce",
#     return_intermediate_steps=False,
#     question_prompt=QUESTION_PROMPT,
#     combine_prompt=COMBINE_PROMPT,
# )
# llm_output = chain({"input_documents": docs, "question": query}, return_only_outputs=True)
# print(llm_output["output_text"])

prompt_template = """Use the following summaries of news articles to create a professional summarized news report for a business executive. It should group relevant information together and be written in a concise, professional manner. Provide a bullet point list of each of the different categories. Come up with relevant categories based on the summaries. Use bullet points for content under each category title. Include categories in decreasing order of importance. Create categories such that there is a maximum of 5 bullet points per category.


News to summarize:
```
{text}
```


EXECUTIVE NEWS SUMMARY, WITH RELEVANT TITLES AND BULLET POINTS:"""
PROMPT_TEMPLATE = PromptTemplate(template=prompt_template, input_variables=["text"])
# prompt_template = """Use the following summaries of news articles to create a professional summarized news report for a business executive. It should group relevant information together under categories, and be written in a concise, professional manner. Provide a bullet point list of each of the different categories. Come up with relevant categories based on the summaries. Use bullet points for content under each category title.

# Use the any of the following categories:
#     - Geopolitics
#     - Business, Finance, and Economics
#     - Technology
#     - Sports
#     - Science and Health
#     - Entertainment
#     - Lifestyle


# {text}


# EXECUTIVE NEWS SUMMARY, WITH GENERALIZED, HIGH-LEVEL TITLES AND BULLET POINTS:"""
# prompt_template = """Use the following summaries of news articles to create a professional summarized news report, in the style of an Executive Summary. It should group relevant information together under categories, and be written in a concise, professional manner. Provide a bullet point list of each of the different categories. Come up with relevant categories based on the summaries. Use bullet points for content under each category title. Try to include as much of the unique, interesting, and impactful information as possible. Always include the country that a particular item is from.

# Only include information that is in the summaries. Do not hallucinate information.

# Exclude any messages about errors in retrieving articles, or bot activity detected.

# Create relevant global categories to group the news.

# Include as much of the unique, interesting, and impactful information as possible. Include the country that a particular item is from. Include news from all over the world. Make it as diverse as possible.

# {text}


# EXECUTIVE SUMMARY OF NEWS, WITH GENERALIZED, HIGH-LEVEL TITLES AND BULLET POINTS:"""

# prompt_template = """Use the following summaries of news articles to create a professional summarized news report, in the style of an Executive Summary.Provide a bullet point list of each of the different categories. Try to include as much of the unique, interesting, and impactful information as possible. Always include the country that a particular item is from.

# Only include information that is in the summaries. Do not hallucinate information.

# Exclude any messages about errors in retrieving articles, or bot activity detected.

# Create relevant global categories to group the news by continent.

# Include the country that a particular item is from. Include news from all over the world. Make it as diverse as possible. Include as much news as possible.


# News:
# {text}


# EXECUTIVE SUMMARY OF NEWS, WITH CATEGORIES AND BULLET POINTS:"""
# combine_prompt = """Create an executive news summary. An informative, desnse, concise, bullet point list of news items from around the world. The news summaries should be diverse and informative.

# News to summarize:
# {text}

# Executive Summary of the News, with bullet points, grouped by a high-level, well-generalized categories:"""

# # Use the any of the following categories:
# #     - Current Events
# #     - People
# #     - Places
# #     - Science and Technology
# #     - Society
# #     - Sports and Recreation
# COMBINE_PROMPT = PromptTemplate(template=combine_prompt, input_variables=["text"])

# map_prompt_template = """Summarize the following news article to a single bullet point. Make it shorthand, and as information dense and concise as possible. Include the most relevant and impactful information from the article. Always include the country that the article is discussing. It must be a single, short sentence.

# Article:

# {text}
# """

# refine_prompt = """Your job is to create a bullet point list of news, grouped by category. You already have a partially-completed list of bullet points, with another news article to summarize and add to the list in a meaningful manner.

# Existing list:
# {existing_answer}

# News to add to the sumamry:
# {text}"""
# REFINE_PROMPT = PromptTemplate(template=refine_prompt, input_variables=["text", "existing_answer"])
# MAP_PROMPT = PromptTemplate(template=map_prompt_template, input_variables=["text"])

chain = load_summarize_chain(
    llm,
    chain_type="map_reduce",
    return_intermediate_steps=False,
    map_prompt=PROMPT_TEMPLATE,
    combine_prompt=PROMPT_TEMPLATE,
    verbose=True,
)

# chain = load_summarize_chain(
#     llm,
#     chain_type="refine",
#     return_intermediate_steps=True,
#     question_prompt=MAP_PROMPT,
#     refine_prompt=REFINE_PROMPT,
#     verbose=True,
# )
llm_result = chain({"input_documents": docs}, return_only_outputs=False)

# pp.pprint(llm_result)
# print()
print(llm_result["output_text"])
