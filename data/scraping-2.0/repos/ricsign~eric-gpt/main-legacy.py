import os
import chainlit as cl
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import PyPDFLoader, TextLoader, NotionDBLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.utilities import SerpAPIWrapper
from langchain.tools import ShellTool
from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings
from chainlit.types import AskFileResponse
from config import open_ai_key, serp_api_key
from langchain.agents import initialize_agent, AgentExecutor
from langchain.agents import AgentType
import tempfile

# Initialize necessary components
os.environ['OPENAI_API_KEY'] = open_ai_key
os.environ['SERPAPI_API_KEY'] = serp_api_key
default_chain = None
llm = ChatOpenAI(temperature=1, streaming=True)
web_search = SerpAPIWrapper()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
embeddings = OpenAIEmbeddings()
shell_tool = ShellTool()

welcome_message = """welcome to eric gpt! choose an option below:"""


@cl.on_chat_start
async def initialize():
    global default_chain
    # Initialize a default chain that only uses ChatOpenAI
    default_chain = RetrievalQAWithSourcesChain.from_chain_type(
        llm,
        chain_type="stuff"
    )
    # Prompt the user with the welcome message
    await cl.Message(content=welcome_message).send()


@cl.on_message
async def handle_choice(message):
    content = message.content.strip()
    if content == '1':
        await prompt_upload()
    elif content == '2':
        await cl.Message(content="You can now type your query.").send()
    else:
        await cl.Message(content=f"Invalid choice. {welcome_message}").send()


async def prompt_upload():
    files = await cl.AskFileMessage(
        content="Please upload your file:",
        accept=["text/plain", "application/pdf"],
        max_size_mb=20,
        timeout=180
    ).send()
    await process_uploaded_file(files[0])


async def process_uploaded_file(file: AskFileResponse):
    msg = await cl.Message(content=f"Processing `{file.name}`...").send()

    if file.type == "text/plain":
        Loader = TextLoader
    elif file.type == "application/pdf":
        Loader = PyPDFLoader

    with tempfile.NamedTemporaryFile() as tmpfile:
        tmpfile.write(file.content)
        loader = Loader(tmpfile.name)
        documents = loader.load()
        docs = text_splitter.split_documents(documents)
        for i, doc in enumerate(docs):
            doc.metadata["source"] = f"source_{i}"

    # Save data in the user session
    cl.user_session.set("docs", docs)

    # Create a unique namespace for the file
    docsearch = Chroma.from_documents(
        docs, embeddings
    )

    chain = RetrievalQAWithSourcesChain.from_chain_type(
        ChatOpenAI(temperature=0, streaming=True),
        chain_type="stuff",
        retriever=docsearch.as_retriever(max_tokens_limit=4097),
    )

    msg.content = f"`{file.name}` processed. You can now ask questions!"
    await msg.update()

    cl.user_session.set("chain", chain)

    # Initialize shell agent
    shell_tool.description = shell_tool.description + f" args {shell_tool.args}".replace("{", "{{").replace("}", "}}")
    agent = initialize_agent(
        [shell_tool],
        llm,
        agent=AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
        handle_parsing_errors=True
    )
    cl.user_session.set("shell_agent", agent)

@cl.on_message
async def main(message):
    if "[notion]" in message:
        query = message.split("[notion]")[1].strip()
        notion_loader = NotionDBLoader()
        docs = notion_loader.load(query)  # Adjust as per the actual implementation
        response_content = "\n".join(f"{doc.title}\n{doc.content}" for doc in docs)
        await cl.Message(content=response_content).send()
    elif "[search]" in message:
        query = message.split("[search]")[1].strip()
        search_results = web_search.run(query)
        await cl.Message(content=search_results).send()
    elif "[shell]" in message:
        shell_command = message.split("[shell]")[1].strip()
        message = shell_command  # Update the message content to only contain the shell command
        shell_agent = cl.user_session.get("shell_agent")  # type: AgentExecutor
        cb = cl.LangchainCallbackHandler(stream_final_answer=True)
        await cl.make_async(shell_agent.run)(message, callbacks=[cb])
    else:
        chain = cl.user_session.get("chain")  # type: RetrievalQAWithSourcesChain
        cb = cl.AsyncLangchainCallbackHandler(
            stream_final_answer=True, answer_prefix_tokens=["FINAL", "ANSWER"]
        )
        cb.answer_reached = True
        res = await chain.call(message, callbacks=[cb])

        answer = res["answer"]
        sources = res["sources"].strip()
        source_elements = []

        docs = cl.user_session.get("docs")
        metadatas = [doc.metadata for doc in docs]
        all_sources = [m["source"] for m in metadatas]

        if sources:
            found_sources = []
            for source in sources.split(","):
                source_name = source.strip().replace(".", "")
                try:
                    index = all_sources.index(source_name)
                except ValueError:
                    continue
                text = docs[index].page_content
                found_sources.append(source_name)
                source_elements.append(cl.Text(content=text, name=source_name))

            if found_sources:
                answer += f"\nSources: {', '.join(found_sources)}"
            else:
                answer += "\nNo sources found"

        if cb.has_streamed_final_answer:
            cb.final_stream.elements = source_elements
            await cb.final_stream.update()
        else:
            await cl.Message(content=answer, elements=source_elements).send()
