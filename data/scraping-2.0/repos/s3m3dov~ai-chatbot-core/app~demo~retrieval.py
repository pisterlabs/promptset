import json
from typing import Optional, List

from langchain.chains import VectorDBQA
from langchain.document_loaders import TextLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma

log = print


class ChatBot:
    def __init__(self):
        self.filename = "../../history.json"
        self.vectordb = self.init_chroma_db()
        self.engine = VectorDBQA.from_chain_type(llm=OpenAI(), chain_type="stuff", vectorstore=self.vectordb)

    def load_process_docs(self) -> List[Document]:
        # Load and process the text
        loader = TextLoader(self.filename)
        documents = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        texts = text_splitter.split_documents(documents)
        log(f"Loaded {len(texts)} documents")

        return texts

    def init_chroma_db(self) -> Chroma:
        texts = self.load_process_docs()
        # Embed and store the texts
        # Supplying a persist_directory will store the embeddings on disk
        persist_directory = 'db'

        embedding = OpenAIEmbeddings()
        vectordb = Chroma.from_documents(documents=texts, embedding=embedding, persist_directory=persist_directory)
        vectordb.persist()

        # Now we can load the persisted database from disk, and use it as normal.
        vectordb = Chroma(persist_directory=persist_directory, embedding_function=embedding)
        log(f"Loaded vectordb")
        return vectordb

    def save_prompt_n_response(self, prompt: str, response: Optional[str]) -> None:
        # Open the JSON file in read mode.
        with open(self.filename, "r") as file:
            # Load the JSON data into a Python object.
            file_obj = file.read()
            if not file_obj:
                data = []
            else:
                data = json.loads(file_obj)

        # Append the new data to the Python object.
        data.append({"prompt": prompt, "response": response})

        # Write the updated JSON data to the file.
        with open(self.filename, "w") as file:
            json.dump(data, file, indent=4)

        log("Saved to database!")
        return


if __name__ == "__main__":
    bot = ChatBot()

    while True:
        prompt = input("Enter a prompt: ")

        if prompt == "exit":
            log("Goodbye!")
            break
        else:
            response = bot.engine.run(prompt)
            log(f"Response: {response}")
            bot.save_prompt_n_response(prompt, response)
