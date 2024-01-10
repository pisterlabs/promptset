from langchain import OpenAI, PromptTemplate
from langchain.docstore.document import Document
from langchain.prompts import PromptTemplate
from langchain.chains.summarize import load_summarize_chain
import tiktoken
import os
from langchain.chains.qa_with_sources import load_qa_with_sources_chain
import pprint

pp = pprint.PrettyPrinter(indent=4)


llm = OpenAI(model_name="text-davinci-003", temperature=0, max_tokens=4000, batch_size=1)

enc = tiktoken.get_encoding("cl100k_base")

files = [f"{n}.txt" for n in range(1, 71)]

# Get the summary files
files = [file for file in os.listdir("summaries") if file.endswith(".txt")]

docs = [Document(page_content=open(f"summaries/{file}").read()) for file in files]

chain = load_qa_with_sources_chain(llm, chain_type="map_reduce")

# prompt_template = """Use the following summaries of news articles to create a professional summarized news report for a business executive. It should group relevant information together and be written in a concise, professional manner. Provide a bullet point list of each of the different categories. Come up with relevant categories based on the summaries. Use bullet points for content under each category title.


# {text}


# EXECUTIVE NEWS SUMMARY, WITH RELEVANT TITLES AND BULLET POINTS:"""
map_prompt_template = """Use the following article summary to create a single bullet point summarizing the key information.

If relevant, preserve the date information. Make the bullet point as information dense as possible.

It is okay to use shorthand, and to drop unnecessary words. Use as few words as possible. Only include any information that relates to the title of the article summary.

The output must be in the format:
```
- <bullet point>
```

The bullet point must not exceed 15 words. It is okay to be grammatically incorrect. The bullet point must be a single sentence.

Article Summary to be converted to a Bullet Point:
```
{text}
```

Complete bullet point for the article summary:"""
MAP_PROMPT = PromptTemplate(template=map_prompt_template, input_variables=["text"])


reduce_prompt_template = """Use the following bullet points discussing news articles to create a concise professional summary, in the style of an executive summary. Try to establish relationships between items.

The executive summary should be a collection of concise bullet points nested undeer relevant titles. Use any dates in the bullet point to make the information chronological.

Use bullet points for content under each related category title. Write the article as in bullet point format. Complete the article before stopping. Combine the gist of any bullet points that are related. Keep the bullet points as concise as possible.

Put each of the bullet points under relevant related category titles. Include a maximum of 3 bullet points per category title in the format:

```
# <category title 1>
- <bullet point 1>
- <bullet point 2>
- <bullet point 3>

# <category title 2>
- <bullet point 1>
- <bullet point 2>
- <bullet point 3>
```

Ignore any content about unusual activity and cookies in the news articles. Ignore any articles that say it could not access the article. It is not relevant to the summary. These are errors and should be ignored. Only give complete bullet points.

Article Bullet Points:

```
{text}
```

# A COMPLETE NEWS EXECUTIVE SUMMARY, IN BULLET POINT LIST WITH ONLY COMPLETE BULLET POINTS:"""
REDUCE_PROMPT = PromptTemplate(template=reduce_prompt_template, input_variables=["text"])


# Running the chain
chain = load_summarize_chain(
    OpenAI(temperature=0),
    chain_type="map_reduce",
    return_intermediate_steps=True,
    map_prompt=MAP_PROMPT,
    combine_prompt=REDUCE_PROMPT,
    verbose=True,
)
llm_result = chain({"input_documents": docs}, return_only_outputs=False)

# pp.pprint(llm_result)
# print()
print(llm_result["output_text"])
