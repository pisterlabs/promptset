from robocorp.tasks import task
from robocorp import vault, workitems, storage

from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import UnstructuredPDFLoader
from langchain.indexes import VectorstoreIndexCreator

import os

@task
def simple_doc_qa():

    # Check if the OpenAI API key exists in the env variables
    # If ont read the it from Control Room Vault, and set.
    if "OPENAI_API_KEY" not in os.environ:
        openai_secrets = vault.get_secret("OpenAI")
        os.environ["OPENAI_API_KEY"] = openai_secrets["key"]

    # Get all PDF files from input work item
    paths = workitems.inputs.current.get_files("*.pdf")
    
    # Create a list of PDF loaders, for each file
    loaders = []
    for path in paths:
        loader = UnstructuredPDFLoader(str(path))
        loaders.append(loader)

    # Create a vector db index (locally) and the llm.
    # Here you would connect to a database service of your choise to persistently store it.
    index = VectorstoreIndexCreator().from_loaders(loaders)
    llm = ChatOpenAI(model="gpt-4")

    # Try your own questions here!
    result = index.query_with_sources("How much money Mr Osborn stole from Laura?", llm=llm)
    print(result)
