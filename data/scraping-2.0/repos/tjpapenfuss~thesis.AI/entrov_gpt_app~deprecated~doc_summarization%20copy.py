from langchain.chains.summarize import load_summarize_chain
from langchain.chat_models import ChatOpenAI
from langchain.text_splitter import CharacterTextSplitter
from langchain.text_splitter import RecursiveCharacterTextSplitter
import config
from langchain import PromptTemplate


def read_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as infile:
        return infile.read()

openai_api_key = config.api_key
llm=ChatOpenAI(openai_api_key=openai_api_key, model_name="gpt-3.5-turbo")
text_splitter = RecursiveCharacterTextSplitter(chunk_size = 3000, chunk_overlap=0, separators=[" ", ",", "\n"])

documents = read_file("areenablingpartnersuccess.txt")
docs = text_splitter.create_documents(documents)
llm3 = ChatOpenAI(temperature=0,
                 openai_api_key=openai_api_key,
                 max_tokens=1000,
                 model='gpt-3.5-turbo'
                )
map_prompt = """
You will be given a section of a website. This section will be enclosed in triple backticks (```)
Your goal is to give a summary of this section so that a reader will have a full understanding of what happened.
Your response should be at least three paragraphs and fully encompass what was said in the passage.

```{text}```
FULL SUMMARY:
"""
map_prompt_template = PromptTemplate(template=map_prompt, input_variables=["text"])

map_chain = load_summarize_chain(llm=llm3,
                             chain_type="stuff",
                             prompt=map_prompt_template)
# Then go get your docs which the top vectors represented.

selected_docs = [docs[doc] for doc in selected_indices]
# Let's loop through our selected docs and get a good summary for each chunk. We'll store the summary in a list.

# Make an empty list to hold your summaries
summary_list = []

# Loop through a range of the lenght of your selected docs
for i, doc in enumerate(selected_docs):
    
    # Go get a summary of the chunk
    chunk_summary = map_chain.run([doc])
    
    # Append that summary to your list
    summary_list.append(chunk_summary)
    
    print (f"Summary #{i} (chunk #{selected_indices[i]}) - Preview: {chunk_summary[:250]} \n")


llm3 = ChatOpenAI(temperature=0,
                 openai_api_key=openai_api_key,
                 max_tokens=3000,
                 model='gpt-3.5-turbo',
                 request_timeout=120
                )
combine_prompt = """
You will be given a series of summaries from a book. The summaries will be enclosed in triple backticks (```)
Your goal is to give a verbose summary of what happened in the story.
The reader should be able to grasp what happened in the book.

```{text}```
VERBOSE SUMMARY:
"""
# Loaders
from langchain.schema import Document
summaries = Document(page_content=texts)
combine_prompt_template = PromptTemplate(template=combine_prompt, input_variables=["text"])

reduce_chain = load_summarize_chain(llm=llm3,
                             chain_type="stuff",
                             prompt=combine_prompt_template,
#                              verbose=True # Set this to true if you want to see the inner workings
                                   )

output = reduce_chain.run([summaries])