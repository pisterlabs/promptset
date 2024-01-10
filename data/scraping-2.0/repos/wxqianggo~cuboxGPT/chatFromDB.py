from langchain import PromptTemplate
from langchain.chains.qa_with_sources import load_qa_with_sources_chain
from langchain.llms import OpenAI
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from rich.console import Console
import tiktoken
console = Console()
directory = "text"
persist_directory = 'db'


def queryFromPersistantDB(query):
    if query is None:
        # query = "I would like a tool to speed up my coding with the help of AI and GPT? In this way I can chat with my code. Give the name of the tool and the url"
        query = "I am looking for a tool to draw grpah from data. Like bar chart, pie chart."
    docsearch = Chroma(persist_directory=persist_directory,
                       embedding_function=OpenAIEmbeddings())
    docs = docsearch.similarity_search(query)
    question_prompt_template = """You are a tech editor. You have reviewd thousands of apps and know a lot about tech and software stuff. Now you are to help people find the things that is suitable for their nedds. Use the following portion of a long document to see if any of the tools is relevant to answer the question.
    Return any relevant text in English.
    {context}
    Question: {question}
    Relevant text, if any, in English:"""
    QUESTION_PROMPT = PromptTemplate(
        template=question_prompt_template, input_variables=[
            "context", "question"]
    )

    combine_prompt_template = """Given the following extracted parts of a long list of tools and their intros and a question, create a final answer with references ("SOURCES" and "URL").
    If you don't know the answer, just say that you don't know. Don't try to make up an answer.
    ALWAYS return "SOURCES" and "URL" part in your answer.
    Respond in English.

    QUESTION: {question}
    =========
    {summaries}
    =========
    FINAL ANSWER:"""
    COMBINE_PROMPT = PromptTemplate(
        template=combine_prompt_template, input_variables=[
            "summaries", "question"]
    )

    chain = load_qa_with_sources_chain(OpenAI(temperature=0), chain_type="map_reduce",
                                       return_intermediate_steps=True, question_prompt=QUESTION_PROMPT, combine_prompt=COMBINE_PROMPT)

    # console.log(
    # chain({"input_documents": docs, "question": query}, return_only_outputs=True))
    # Choose the encoding for the OpenAI model you're working with
    encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")

    # Encode the text
    tokens = encoding.encode("\n".join([i.page_content for i in docs]))
    while len(tokens) > 3500:
        docs = docs[: -1]
        tokens = encoding.encode("\n".join([i.page_content for i in docs]))

    response = chain({"input_documents": docs, "question": query},
                     return_only_outputs=True)
    console.log(response["output_text"])
    # console.log(response)


if __name__ == "__main__":
    # takes input from CLI arguments
    queryFromPersistantDB()
