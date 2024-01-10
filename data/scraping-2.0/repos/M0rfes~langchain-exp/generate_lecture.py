import os
from langchain import PromptTemplate
from langchain.vectorstores import Pinecone
from langchain.llms import OpenAI
from langchain.chains import LLMChain
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains.summarize import load_summarize_chain
from langchain.chains import AnalyzeDocumentChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
import pinecone

open_api_key = "<open-ai key>"

os.environ['OPENAI_API_KEY'] = open_api_key

pinecone.init(
    api_key="<pincone-key>",
    environment="<pincone env>",
)

llm = OpenAI(openai_api_key=open_api_key,
             temperature=0.5, frequency_penalty=0.7, presence_penalty=0.3)
embedding = OpenAIEmbeddings(openai_api_key=open_api_key)


prompt_template = """Use the provided Context and topic to generate a script for a pre-recoded video Lecture. The script should be 5000 charters long.
    Context: {context}
    Topic: {topic}
    script:"""

PROMPT = PromptTemplate(
    template=prompt_template, input_variables=["context", "topic"]
)

chain = LLMChain(llm=llm, prompt=PROMPT)

docsearch = Pinecone.from_existing_index(
    "chemistry-2e", embedding=embedding)
docs = docsearch.similarity_search(
    "chapter 5")
inputs = [{"context": doc.page_content, "topic": "chapter 5"}
          for doc in docs]


scripts = " ".join([t['text'] for t in chain.apply(inputs)])

REFINE_PROMPT_TMPL = (
    "Your job is to merge the scripts\n"
    "We have provided an existing script up to a certain point: {existing_answer}\n"
    "Remove the <EOS> token from the end of the script\n"
    "We have the opportunity to refine the existing script"
    "and add to it (only if needed) with some more context below.\n"
    "------------\n"
    "{text}\n"
    "------------\n"
    "Given the new context, combine both the scripts\n"
    "If the context isn't useful, return the original script."
    "If you are done with the script add <EOS> to the end of the script"
)
REFINE_PROMPT = PromptTemplate(
    input_variables=["existing_answer", "text"],
    template=REFINE_PROMPT_TMPL,
)


question_template = """Add to the script if needed:


"{text}"


script:"""
QUESTION_PROMPT = PromptTemplate(
    template=question_template, input_variables=["text"])

summary_chain = load_summarize_chain(llm, chain_type="refine",
                                     question_prompt=QUESTION_PROMPT,
                                     refine_prompt=REFINE_PROMPT,
                                     )

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=5000,
    chunk_overlap=0,
)

summarize_document_chain = AnalyzeDocumentChain(
    combine_docs_chain=summary_chain,
    text_splitter=text_splitter,
)


final_script = summarize_document_chain.run(scripts)

completion_template = """Your job is to add to a partially generated script.
    You have been provided with partial script: {script}.
    You have the opportunity to add to it and finish the script with some more 
    Context: {context}
    This is the topic of the script: {topic}
    Add to the script if needed else return the original script.
    Add <EOS> at the end the script.
    """

completion_PROMPT = PromptTemplate(
    template=completion_template, input_variables=[
        "context", "topic", "script"]
)

context = " ".join([doc.page_content
                    for doc in docs])

with open("lecture.txt", "w") as f:
    print("generating script")
    print(final_script)
    chain = LLMChain(llm=llm, prompt=completion_PROMPT)
    while not final_script.endswith("<EOS>"):
        last_script = chain.apply([
            {"context": context, "topic": "chapter 5",
             "script": final_script}
        ])[0]['text']
        final_script += last_script
        print(last_script)
    else:
        f.write(final_script)
