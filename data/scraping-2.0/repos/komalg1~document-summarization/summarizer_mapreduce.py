import os
from azure.identity import DefaultAzureCredential
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains.summarize import load_summarize_chain
from langchain.chat_models import AzureChatOpenAI
from langchain.document_loaders import TextLoader
import openai
import textwrap

class Summarizer:
    def __init__(self):
        #self.api_version = '2023-08-01-preview'
        self.openai_deploymentname = 'DEPLOYMENT_NAME'
        self.azure_endpoint = f'https://{self.openai_deploymentname}.openai.azure.com/openai'
        self.credential = DefaultAzureCredential()
        
        os.environ["OPENAI_API_TYPE"] = "azure"
        os.environ["OPENAI_API_VERSION"] = "2023-08-01-preview"
        os.environ["OPENAI_API_BASE"] = self.azure_endpoint 
        os.environ["OPENAI_API_KEY"] = self.credential.get_token("https://cognitiveservices.azure.com/.default").token

        openai.api_type = "azure"
        openai.api_base = self.azure_endpoint 
        openai.api_version = "2023-08-01-preview"
        openai.api_key = self.credential.get_token("https://cognitiveservices.azure.com/.default").token
            
    def load_document(self):
        cwd = os.getcwd()
        loader = TextLoader(f'{cwd}/how_to_win.txt')
        documents = loader.load()
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        docs = text_splitter.split_documents(documents)
        return docs[:6]
        
    def summarize_mapreduce_summary_version1(self):
        
        llm = AzureChatOpenAI(openai_api_base=self.azure_endpoint,
                openai_api_version="2023-08-01-preview",
                deployment_name='gpt-35-turbo',
                openai_api_key=self.credential.get_token("https://cognitiveservices.azure.com/.default").token,
                openai_api_type = "azure",
                max_tokens=2500)
        self.text = self.load_document()
        
        chain = load_summarize_chain(llm=llm, chain_type="map_reduce")
        output_summary = chain.run(self.text)
        wrapped_text = textwrap.fill(output_summary, width=100)
        print(wrapped_text)
        
if __name__ == '__main__':
    summarizer = Summarizer()
    summary = summarizer.summarize_mapreduce_summary_version1()