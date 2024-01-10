import os
import sqlite3
import click
from rich.console import Console
from rich.markdown import Markdown
from pathlib import Path
from .config import get_or_create_config_path

# import openai

DOCS_VERSION_NUMBER = '0.0.3'
@click.group()
@click.version_option(DOCS_VERSION_NUMBER, message='docs version: %(version)s')
def cli():
    """
    A CLI for conversational retrieval using langchain and OpenAI.

    QUERY is the initial question to start the conversation.
    """
    pass


@click.command(help='The initial question to start the conversation.')
@click.argument('query', required=True )
def ask(query):
    from langchain.chains import ConversationalRetrievalChain
    from langchain.chat_models import ChatOpenAI
    from langchain.embeddings import OpenAIEmbeddings
    from langchain.indexes.vectorstore import VectorStoreIndexWrapper
    from langchain.vectorstores import Chroma
    from langchain.llms import OpenAI

   
    config_path = os.path.expanduser('~/.config/docs/yt-fts')

    if not os.path.exists(config_path):
        print("No datasets found. use load command to load a dataset.")
        return
    

    # vectorstore = Chroma(persist_directory=config_path, embedding_function=OpenAIEmbeddings())
    # index = VectorStoreIndexWrapper(vectorstore=vectorstore)

    # chain = ConversationalRetrievalChain.from_llm(
    #     llm=ChatOpenAI(model="gpt-3.5-turbo"),
    #     retriever=index.vectorstore.as_retriever(search_kwargs={"k": 1}),
    # )


    vectorstore = Chroma(persist_directory=config_path, embedding_function=OpenAIEmbeddings())
    vectordbkwargs = {"search_distance": 0.9}

    chain = ConversationalRetrievalChain.from_llm(
        OpenAI(temperature=0), vectorstore.as_retriever(), return_source_documents=True
    )
    chat_history = []
    result = chain(
        {"question": query, "chat_history": chat_history, "vectordbkwargs": vectordbkwargs}
        )

    # chat_history = []
    console = Console()
    while True:
        if not query:
            # query = input("Prompt: ")
            query = click.prompt('Prompt', type=str)
        if query in ['quit', 'q', 'exit']:
            # click.echo("Exiting...")
            console.print("Exiting...")
            break

        # print(chain, type(chain))
        result = chain({"question": query, "chat_history": chat_history})
        print(result["source_documents"][0])

        console.print("[bold]Answer:[/bold] " + result['answer'])
        chat_history.append((query, result['answer']))
        query = None



@click.command(help='The initial question to start the conversation.')
@click.argument('query', required=True )
def search(query):
    from langchain.chains import ConversationalRetrievalChain
    from langchain.embeddings import OpenAIEmbeddings
    from langchain.vectorstores import Chroma
    from langchain.llms import OpenAI

   
    config_path = os.path.expanduser('~/.config/docs/yt-fts')

    if not os.path.exists(config_path):
        print("No datasets found. use load command to load a dataset.")
        return
    
    vectorstore = Chroma(persist_directory=config_path, embedding_function=OpenAIEmbeddings())
    vectordbkwargs = {"search_distance": 0.9}

    chain = ConversationalRetrievalChain.from_llm(
        OpenAI(temperature=0), vectorstore.as_retriever(), return_source_documents=True
    )
    chat_history = []
    result = chain(
        {"question": query, "chat_history": chat_history, "vectordbkwargs": vectordbkwargs}
        )

    result = chain({"question": query, "chat_history": chat_history})
    document = result["source_documents"][0].metadata["source"]
    page_content = result["source_documents"][0].page_content

    console = Console()
    console.print(f"[bold]Document:[/bold] {document}\n")
    console.print(f"[bold]Page Content:[/bold]{page_content}\n")   
    console.print("[bold]Answer:[/bold]\n" + result['answer'])



@click.command(help='Load a dataset from a directory')
@click.argument('data_path', required=True)
def load(data_path):

    from langchain.document_loaders import DirectoryLoader
    from langchain.document_loaders import UnstructuredFileLoader

    from langchain.indexes import VectorstoreIndexCreator
    persistant_path = os.path.join(get_or_create_config_path(), "yt-fts")

    if Path(data_path).is_dir():
        loader = DirectoryLoader(data_path)
    
    if Path(data_path).is_file():
        loader = UnstructuredFileLoader(data_path) 

    index = VectorstoreIndexCreator(vectorstore_kwargs={
        "persist_directory": persistant_path 
        }).from_loaders([loader])
    

@click.command(help='List all loaded datasets')
def list():

    from rich.table import Table
    config_path = os.path.expanduser('~/.config/docs/yt-fts')

    if not os.path.exists(config_path):
        print("No datasets found. use load command to load a dataset.")
        return    

    conn = sqlite3.connect(os.path.join(config_path, "chroma.sqlite3"))
    curr = conn.cursor()
    curr.execute("""
                 SELECT DISTINCT string_value FROM embedding_metadata 
                 WHERE key ='source' 
                 """)
    documents = curr.fetchall() 

    console = Console()

    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Documents")

    for document in documents:
        table.add_row(Path(document[0]).name)
    
    console.print(table)




    


cli.add_command(search)
cli.add_command(ask)
cli.add_command(load)
cli.add_command(list)