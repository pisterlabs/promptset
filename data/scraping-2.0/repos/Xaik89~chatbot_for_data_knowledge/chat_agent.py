
from glob import iglob
import jq
from tqdm import tqdm
import os
import json
import logging

from langchain.agents.agent_toolkits import JiraToolkit
from langchain.utilities import JiraAPIWrapper
from langchain.agents import Tool
from langchain.memory import ConversationBufferMemory
from langchain.chat_models import ChatOpenAI
from langchain.agents import initialize_agent
from langchain.docstore.document import Document
from langchain.document_loaders import ConfluenceLoader
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.schema import SystemMessage

# Set up the logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


prompt_summarize_g_chats = """Group Name: Chat History Summary

Introduction:
This chat history summary aims to provide a concise and comprehensive overview of the discussions. The summary will capture the main points and key details of the text, ensuring it is well-organized and easy to read with clear headings and subheadings. The length of the summary will be appropriate to capture the main points and key details without including unnecessary information or becoming overly long. The main bullets will not be repeated, and the response will conclude with a conclusion.

Summary:

I. Heading: Discussion Topic 1

Key Point 1: [Briefly summarize key point 1]
Key Point 2: [Briefly summarize key point 2]
Key Point 3: [Briefly summarize key point 3]
II. Heading: Discussion Topic 2

Key Point 1: [Briefly summarize key point 1]
Key Point 2: [Briefly summarize key point 2]

III. Heading: Discussion Topic 3

Key Point 1: [Briefly summarize key point 1]
Key Point 2: [Briefly summarize key point 2]
Key Point 3: [Briefly summarize key point 3]
Key Point 4: [Briefly summarize key point 4]
IV. Heading: Discussion Topic 4

Key Point 1: [Briefly summarize key point 1]
Key Point 2: [Briefly summarize key point 2]
Key Point 3: [Briefly summarize key point 3]

Conclusion:
[Conclude the summary with a brief conclusion]"""

PREFIX = """As an agent, your role is to utilize specific tools to gather accurate information and provide reliable and trustworthy answers. In order to ensure the highest quality response, you must adhere to the following rules:

Rule 1: Answer the questions to the best of your ability based on the observations provided.
Rule 2: Strictly rely on the information available within the given observations and avoid using any external sources.
Rule 3: Refrain from making assumptions or drawing conclusions unless the observation explicitly provides all the necessary information for the final answer.
Rule 4: You have access to almost all the information required to answer the questions, but you need to find the appropriate context using the tools at your disposal.
Rule 5: The tools available for you to retrieve answers are "confluence wiki", "google chats" and "jira".
Rule 6: Present your final thoughts in a structured manner.

You have access to the following tools: "confluence wiki" for obtaining general information related to research,
 engineering, and other relevant topics, "google chats" for obtaining updated information about clients and their requests,
  and "jira" for retrieving information about tasks and their current status."""
FORMAT_INSTRUCTIONS = """Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question in structured way and without using word "part" """
SUFFIX = """Begin!

Question: {input}
Thought:{agent_scratchpad}"""


class VectorDBAtlassin:
    def __init__(self, model_name=None, force_reload=False):

        embeddings = OpenAIEmbeddings(openai_api_key=os.environ.get("OPENAI_API_KEY"))

        # Create/load databases independently
        self.vector_store_wiki = self.load_database(
            "./chroma_db_confluence",
            "confluence",
            embeddings,
            force_reload,
            self.load_confluence_wiki)

        self.vector_store_g_chats = self.load_database(
            "./chroma_db_g_chats",
            "g_chats-summary",
            embeddings,
            force_reload,
            lambda: self.load_g_chat_backup(model_name))

        self.vector_store_wiki.persist()
        self.vector_store_g_chats.persist()

    @staticmethod
    def load_database(directory, collection_name, embeddings, force_reload, load_documents_func):
        if not force_reload and os.path.exists(directory):
            logger.info(f"Loading from disk vector db: {collection_name}")
            return Chroma(persist_directory=directory, embedding_function=embeddings, collection_name=collection_name)
        else:
            logger.info(f"Loading {collection_name}")
            documents = load_documents_func()
            logger.info(f"Creating vector db for {collection_name}")
            vector_store = Chroma.from_documents(
                documents,
                embeddings,
                collection_name=collection_name,
                persist_directory=directory
            )
            logger.info(f"Done creating vector db for {collection_name}")
            return vector_store

    @staticmethod
    def load_confluence_wiki():
        loader = ConfluenceLoader(
            url=os.environ.get("CONFLUENCE_WIKI_URL"),
            username=os.environ.get("CONFLUENCE_API_USER"),
            api_key=os.environ.get("CONFLUENCE_API_TOKEN"),
        )
        wiki_documents = loader.load(space_key="Wiki", include_attachments=False, limit=10, max_pages=100)
        research_docs = loader.load(space_key="ATW", include_attachments=False, limit=10, max_pages=100)
        customer_docs = loader.load(space_key="CS", include_attachments=False, limit=10, max_pages=100)
        engineering_docs = loader.load(space_key="EN", include_attachments=False, limit=10, max_pages=100)
        product_docs = loader.load(space_key="PS", include_attachments=False, limit=10, max_pages=100)
        policy_docs = loader.load(space_key="POL", include_attachments=False, limit=10, max_pages=100)

        documents = wiki_documents + research_docs + customer_docs + engineering_docs + product_docs + policy_docs

        # clean docs
        documents = [d for d in documents if d.page_content != "" and "pages" in d.metadata["source"] and "OLD" not in d.metadata["source"]]

        # split it into chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=50, add_start_index=True)
        wiki_documents = text_splitter.split_documents(documents)

        VectorDBAtlassin._append_title(wiki_documents[0], part_index=1)
        part_index = 2
        for current_index in range(1, len(wiki_documents)):
            if VectorDBAtlassin.is_same_document(wiki_documents[current_index], wiki_documents[current_index - 1]):
                part_index += 1
            else:
                part_index = 1
            VectorDBAtlassin._append_title(wiki_documents[current_index], part_index)

        return wiki_documents

    @staticmethod
    def _append_title(document, part_index):
        def _get_start_title(document):
            if "ATW" in document.metadata["source"]:
                start_title_option = "Research from AI/Algo team: "
            elif "CS" in document.metadata["source"]:
                start_title_option = "Customer Success (support for Clients): "
            elif "EN" in document.metadata["source"]:
                start_title_option = "Engineering: "
            elif "PS" in document.metadata["source"]:
                start_title_option = "Product: "
            elif "POL" in document.metadata["source"]:
                start_title_option = "Policies & Procedures: "
            else:
                start_title_option = "General Wiki about Edgify: "
            return start_title_option

        start_title = _get_start_title(document)
        document.page_content = f"{start_title}{document.metadata['title']} part{part_index}\n{document.page_content}"

    @staticmethod
    def is_same_document(doc1, doc2):
        return doc1.metadata["id"] == doc2.metadata["id"]

    def load_g_chat_backup(self, model_name, summarize=False):
        final_docs = []
        main_dir = os.environ.get("G_CHATS_BACKUP_PATH")
        logger.info(f"Len of main dir: {len(os.listdir(main_dir))}")
        for space_name in tqdm(os.listdir(main_dir)):
            file_list = [f for f in
                         iglob(f'{main_dir}/{space_name}/**',
                               recursive=True) if os.path.isfile(f) and f.endswith('messages.json')]

            # get metadata
            first_file = file_list[0]
            if "Topic" in first_file:
                metadata_file = "/".join(first_file.split("/")[:-2]) + "/group_info.json"
            else:
                metadata_file = first_file.replace("messages.json", "group_info.json")
            with open(metadata_file) as json_file:
                data = json.load(json_file)
                group_name = data["name"]

            text = f"group name for this talk: {group_name}\n"
            for file_name in file_list:
                with open(file_name) as json_file:
                    data = json.load(json_file)
                    jq_filter = '.messages[] | {text: .text, created_date: .created_date}'
                    dicts = jq.jq(jq_filter).transform(data, multiple_output=True)

                    for d in dicts:
                        if d["created_date"] is None or "2023" not in d["created_date"] or d["text"] in ["", None]:
                            continue
                        text += d["text"] + "\n"

            new_doc = Document(page_content=text, metadata={"group name": group_name})
            # remove empty documents
            final_docs.append(new_doc)

        if summarize:
            final_docs = self.summarize_g_chats(final_docs, model_name)

        # split it into chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=100)
        final_docs = text_splitter.split_documents(final_docs)
        return final_docs

    @staticmethod
    def summarize_g_chats(documents, model_name):
        llm = ChatOpenAI(temperature=0.1, model_name=model_name)
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=3000, chunk_overlap=50)

        for doc in tqdm(documents):
            num_tokens = llm.get_num_tokens_from_messages([SystemMessage(content=doc.page_content)])
            if num_tokens > 8000:
                sub_docs = text_splitter.split_documents([doc])
            else:
                sub_docs = [doc]
            summary = ""
            for inner_doc in sub_docs:
                if summary == "":
                    current_prompt = prompt_summarize_g_chats + inner_doc.page_content
                else:
                    current_prompt = prompt_summarize_g_chats + summary.content + "\n" + inner_doc.page_content
                summary = llm([SystemMessage(content=current_prompt)])

            doc.page_content = summary.content

        return documents

    def get_wiki_db(self):
        return self.vector_store_wiki

    def get_g_chat_db(self):
        return self.vector_store_g_chats


class ChatBot:
    def __init__(self, memory, agent_chain):
        self.memory = memory
        self.agent = agent_chain


def create_chatbot(model_name, seed_memory=None):
    llm = ChatOpenAI(temperature=0, model_name=model_name)
    db_search = VectorDBAtlassin(model_name=model_name)

    jira = JiraAPIWrapper()
    toolkit = JiraToolkit.from_jira_api_wrapper(jira)

    qa_wiki = RetrievalQA.from_chain_type(llm=llm,
                                          chain_type="stuff",
                                          retriever=db_search.get_wiki_db().as_retriever())

    qa_chats = RetrievalQA.from_chain_type(llm=llm,
                                           chain_type="stuff",
                                           retriever=db_search.get_g_chat_db().as_retriever())

    tools = [
        Tool(
            name="confluence wiki",
            func=qa_wiki.run,
            description="useful for all questions that asks about our confluence wiki or any information that can be found regarding Company Edgify",
        ),
        Tool(
            name="google chats",
            func=qa_chats.run,
            description="usefull to get updated info about what is going on in the company and in particular for each client of Edgify",
        ),
        ]

    tools += toolkit.get_tools()

    memory = seed_memory if seed_memory is not None else ConversationBufferMemory(memory_key="chat_history")
    agent_chain = initialize_agent(tools, llm, verbose=True, memory=memory, max_iterations=3,
                                   agent_kwargs={
                                        'prefix': PREFIX,
                                        'format_instructions': FORMAT_INSTRUCTIONS,
                                        'suffix': SUFFIX
                                    })

    return ChatBot(memory, agent_chain)
