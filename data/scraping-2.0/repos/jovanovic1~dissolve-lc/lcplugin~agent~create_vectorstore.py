import os
from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter, Language
import tiktoken
import hashlib
import sys
sys.path.append('.')

class CreatorVectorStore:
    def __init__(self, file_path: str, url: str):
        self.file_path = file_path
        self.url = url

    def encode_url_to_int(self, url_string):
        # Using SHA-256 hash algorithm
        sha256_hash = hashlib.sha256(url_string.encode()).hexdigest()

        # Convert the hexadecimal hash to an integer
        encoded_int = int(sha256_hash, 16)

        return encoded_int
    
    def create_vectorstore(self):
        load_dotenv()
        openai_api_key = os.getenv('OPENAI_API_KEY')

        with open(self.file_path, "r") as f:
            content = f.read()

        tokenizer = tiktoken.get_encoding('cl100k_base')

        text_splitter = RecursiveCharacterTextSplitter.from_language(
            chunk_size=20000,
            chunk_overlap=2000,
            language=Language.HTML,
        )

        texts = text_splitter.create_documents([content])

        llm = ChatOpenAI(
            openai_api_key=openai_api_key,
            model_name='gpt-3.5-turbo-16k-0613',
            temperature=0.0
        )

        template = """
            You are an intelligent HTML understanding agent. You will be given a HTML snippet and you have to summarise the set of actions that user can take of UI elements within the HTML code. Keep the summary concise and restrict answer to the input HTML.

            Here's an example: 

            Sample input HTML: 
            <a href="https://www.logitech.com/en-in/products/tablet-keyboards.html" aria-label="Tablet Keyboards" class="nav-item-link js-nav-item-link " data-style="standard" target="_self" data-analytics-title="tablet-keyboards">
                <span class="default">Tablet Keyboards</span>
            </a>
            Sample output: - Action: Click "Tablet Keyboards" to navigate to Logitech's tablet keyboards page

            Input HTML:
        """ 

        # Iterate over the list 'texts'
        for i in range(len(texts)):
            # Store the original HTML code in the metadata
            texts[i].metadata = {'HTML': texts[i].page_content}

            # Prepare the prompt by appending the page_content to the template
            prompt = template + texts[i].page_content

            # Call llm function with the prepared prompt
            output = llm.predict(prompt)

            # Replace the page_content in the object with the output of llm
            texts[i].page_content = output

        embeddings = OpenAIEmbeddings()

        persist_directory = 'vectordb'
        converted_url = ''.join(filter(str.isalnum, self.url))

        Chroma.from_documents(documents=texts, 
                               embedding=embeddings,
                               collection_name=converted_url,
                               persist_directory=persist_directory)
