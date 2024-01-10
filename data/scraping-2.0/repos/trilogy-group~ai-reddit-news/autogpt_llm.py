import os

import faiss
from dotenv import load_dotenv
from langchain import FAISS, InMemoryDocstore
from langchain.chat_models import ChatOpenAI

load_dotenv()
from langchain.utilities import GoogleSearchAPIWrapper
from langchain.agents import Tool
from langchain.tools.file_management.write import WriteFileTool
from langchain.tools.file_management.read import ReadFileTool
from langchain.embeddings import OpenAIEmbeddings
from langchain_experimental.autonomous_agents import AutoGPT


class PostRatingLLM:
    RATING_TEMPLATE = """Evaluate the following Reddit post based on the following criteria:
    
    1. Does the post provide valuable information or resources that could help someone become an expert in AI?
    2. Does the post contain the latest developments or updates in the field of AI and Language Learning Models (LLMs)?
    3. Would the post be interesting and useful to anyone aspiring to become an expert in AI, regardless of whether they are a developer or not?
    
    Please rate the post on a scale of 1-10 for each criterion, with 1 being 'not at all' and 10 being 'extremely'
    
    Post Title: {post_title}
    Post Body: {post_body}
    Post Comments: {post_comments}
    
    Your final output should only be a single integer rating.
    """

    def __init__(self):
        self._set_llm()

    def _set_llm(self):
        self.agent = AutoGPT.from_llm_and_tools(
            ai_name="Jim",
            ai_role="Assistant",
            tools=self._get_tools(),
            llm=ChatOpenAI(model="gpt-4", temperature=0),
            memory=self._get_db().as_retriever(),
        )
        if os.environ.get("DEBUG") and os.environ.get("DEBUG").lower() == "true":
            self.agent.chain.verbose = True

    @staticmethod
    def _get_tools():
        search = GoogleSearchAPIWrapper()
        return [
            Tool(
                name="search",
                func=search.run,
                description="Useful for when you need to answer questions about current events.",
                return_direct=True,
            ),
            WriteFileTool(),
            ReadFileTool(),
        ]

    @staticmethod
    def _get_db():
        embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
        embedding_size = 1536
        index = faiss.IndexFlatL2(embedding_size)
        return FAISS(embeddings.embed_query, index, InMemoryDocstore({}), {})
        # dataset_path = os.environ.get("DEEPLAKE_DATASET_PATH")
        # return DeepLake(dataset_path=dataset_path, embedding_function=embeddings)

    def rate(self, post_title, post_body, post_comments):
        return self.agent.run(
            [
                self.RATING_TEMPLATE.format(
                    post_title=post_title,
                    post_body=post_body,
                    post_comments=post_comments,
                )
            ]
        )
