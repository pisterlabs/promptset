import os

from langchain import LLMChain, PromptTemplate
from langchain.document_loaders import AzureBlobStorageFileLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.tools import BaseTool
from prompt.QuestionAndAnswerPrompt import QUESTION_AND_ANSWER_PROMPT
from llm.llm import llm


class QuestionAndAnswerTool(BaseTool):
    name = "QuestionAndAnswerTool"
    description = """Useful tool to answer questions related to article content for example, 
    1. introduce the research background of this paper, 
    2. introduce the problems to be solved
    3.the technical route, etc.
    parameter: information: The message should include the file name and the user's question format ('user qustion','file_name') such as (The research background of this article,demo.pdf)
                    """

    def _run(self, information: str):
        conn_string = os.getenv("STORAGE_CONNECTION_STRING",
                                "DefaultEndpointsProtocol=https;AccountName=cggptsc;AccountKey=rWvHP0XV8ji7QnVDDASpbjApgiixQ/RITbzlF62z7CWPkIXWzi6W5ZJIlf0UXU5/Eg5UTwx13XaB+AStuckFbQ==;EndpointSuffix=core.windows.net")
        container_name = os.getenv("CONTAINER_NAME", "gptfiles")
        information = information.replace("(", "").replace(")", "").replace("'", "").replace(" ", "").replace("\"","").split(",")
        question = information[0]
        file_name = information[1]
        prompt = PromptTemplate.from_template(QUESTION_AND_ANSWER_PROMPT)
        chain = LLMChain(llm=llm, prompt=prompt, verbose=True)
        blob_loader = AzureBlobStorageFileLoader(conn_str=conn_string, container=container_name,
                                                 blob_name=file_name)
        splitter = CharacterTextSplitter(chunk_size=2000, chunk_overlap=0)
        documents = blob_loader.load_and_split(splitter)
        print("==============================")
        results = []
        for document in documents:
            result = chain.run(question=question, document=document.page_content)
            results.append(result)
        final_result = chain.run(question=question, document=str(results))
        return final_result

    async def _arun(self, file_name: str):
        pass
