from langchain.prompts import PromptTemplate
from langchain.chains.question_answering import load_qa_chain
from langchain.llms.openai import OpenAI
from langchain.output_parsers import RegexParser
from pages.tools.extract_file import get_document_embeddings
from pages.tools.extract_video import get_video_embeddings
from dotenv import load_dotenv
load_dotenv()


def create_chain():
    # Creating a Q&A Chain using LangChain
    prompt_template = """
    Only use the following pieces of context to answer the question at the end.
    If you don't know the answer, just say:
    "I don't know, information not in the context.",
    don't try to make up an answer.

    This should be in the following format:

    Question: [question here]
    Helpful Answer: [answer here]
    Score: [score between 0 and 100]

    Begin!

    Context:
    ---------
    {context}
    ---------
    Question: {question}
    Helpful Answer:
    """
    # Parse the output
    output_parser = RegexParser(
        regex=r"(.*?)\n[\s]*Score: (.*)",
        output_keys=["answer", "score"],
    )

    # Creat the chain
    PROMPT = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"],
        output_parser=output_parser
    )
    chain = load_qa_chain(OpenAI(temperature=0),
                          chain_type="map_rerank",
                          return_intermediate_steps=True, prompt=PROMPT)
    return chain


def get_response(text, query, flag):
    if flag == "document":
        relevant_chunks = get_document_embeddings(text).similarity_search_with_score(query)
    if flag == "video":
        relevant_chunks = get_video_embeddings(text).similarity_search_with_score(query)
    chunk_docs = []
    for chunk in relevant_chunks:
        chunk_docs.append(chunk[0])
    chain = create_chain()
    results = chain({"input_documents": chunk_docs, "question": query})
    text_reference = ""
    for i in range(len(results["input_documents"])):
        text_reference += results["input_documents"][i].page_content
    output = {"Answer": results["output_text"]}
    return output
