from langchain.prompts import PromptTemplate


from langchain_google_genai import ChatGoogleGenerativeAI
import os
from dotenv import load_dotenv,find_dotenv
_=load_dotenv(find_dotenv())
llm = ChatGoogleGenerativeAI(model="gemini-pro",google_api_key=os.getenv('GOOGLE_API_KEY'),temperature=0)

from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableBranch

chain = (
    PromptTemplate.from_template(
        """Given the user question below, classify it as either being about `linkedin`, `github`, or `Other`.

Do not respond with more than one word.

<question>
{question}
</question>

Classification:"""
    )
    | llm
    | StrOutputParser()
)


linkedin_template = """You are a very smart ai Assistant. \
You are great at answering questions about linkedin and easy to understand manner. \
1.Make analysis of provided context.\
2.Thoroughly review the context.\
3.Understand what is question asking try to relate it to linkedin context.\
4.Check if infoemation is available in context.\
5.For you answer on the basis of context.\
6.If the information is not mentioned in context.Then answer that you don't know.\

Context:
{context}

Here is a question:
{question}"""
linkedin_chain = (PromptTemplate.from_template(linkedin_template)|llm)

github_template = """You are a very very smart ai Assistant. You are great at answering github questions. \
1.Make analysis of provided context.\
2.Thoroughly review the context.\
3.Understand what is question asking try to relate it to github context.\
4.Check if infoemation is available in context.\
5.For you answer on the basis of context.\
6.If the information is not mentioned in context.Then answer that you don't know.\

Context:
{context}
Here is a question:
{question}

Your Response:"""
github_chain = (PromptTemplate.from_template(github_template)|llm)

general_chain =( PromptTemplate.from_template(
    "You are a helpful assistant. Answer the question as accurately as you can.\n\n{question}"
)|llm)

from langchain_core.runnables import RunnableBranch

branch = RunnableBranch(
    (lambda x: "linkedin" in x["topic"].lower(), linkedin_chain),
    (lambda x: "github" in x["topic"].lower(), github_chain),
    general_chain,
)

from langchain.document_loaders import AsyncHtmlLoader

from langchain.document_transformers import Html2TextTransformer
from langchain.document_loaders import WebBaseLoader
from langchain.document_loaders import SeleniumURLLoader
from langchain.document_loaders import PlaywrightURLLoader

full_chain = {"topic": chain, "question": lambda x: x["question"],"context":lambda x: x["context"]} | branch

def social_profile(user_message):
    res=chain.invoke({"question": user_message})
    print(res)
    if res=='github':
        urls = ["https://github.com/SUMEG-04"]
        loader = AsyncHtmlLoader(urls)
        docs = loader.load()

        html2text = Html2TextTransformer()
        docs_transformed = html2text.transform_documents(docs)
        context = str(docs_transformed[0].page_content[0:4000])
        context = context.replace("\n", " ")
        return full_chain.invoke({"question": user_message,"context":context})
    if res=='linkedin':
        urls = ["https://www.linkedin.com/in/sumeg-sharnagat-051851204/"]
        loader = SeleniumURLLoader(urls=urls)
        docs = loader.load()
        print(docs)
        html2text = Html2TextTransformer()
        docs_transformed = html2text.transform_documents(docs)
        context = str(docs_transformed[0].page_content[0:4000])
        context = context.replace("\n", " ")
        print(context)
        #return full_chain.invoke({"question": user_message,"context":context})
        
    
print(social_profile("how many linkedin notification i have?"))