import os
from azure.identity import DefaultAzureCredential
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains.summarize import load_summarize_chain
from langchain.chat_models import AzureChatOpenAI
from langchain.document_loaders import TextLoader
from langchain.prompts import PromptTemplate
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
    
    def summary_refine(self):
        docs = self.load_document()
        prompt_template = """Write a concise summary of the following extracting the key information:


        {text}


        CONCISE SUMMARY:"""
        PROMPT = PromptTemplate(template=prompt_template, 
                                input_variables=["text"])

        refine_template = (
            "Your job is to produce a final summary\n"
            "We have provided an existing summary up to a certain point: {existing_answer}\n"
            "We have the opportunity to refine the existing summary"
            "(only if needed) with some more context below.\n"
            "------------\n"
            "{text}\n"
            "------------\n"
            "Given the new context, refine the original summary"
            "If the context isn't useful, return the original summary."
        )
        refine_prompt = PromptTemplate(
            input_variables=["existing_answer", "text"],
            template=refine_template,
        )
        chain = load_summarize_chain(AzureChatOpenAI(openai_api_base=self.azure_endpoint,
                openai_api_version="2023-08-01-preview",
                deployment_name='gpt-35-turbo',
                openai_api_key=self.credential.get_token("https://cognitiveservices.azure.com/.default").token,
                openai_api_type = "azure"), 
                                    chain_type="refine", 
                                    return_intermediate_steps=True, 
                                    question_prompt=PROMPT, 
                                    refine_prompt=refine_prompt)
        output_summary = chain({"input_documents": docs}, return_only_outputs=True)
        wrapped_text = textwrap.fill(output_summary['output_text'], 
                             width=100,
                             break_long_words=False,
                             replace_whitespace=False)
        print(wrapped_text)

if __name__ == '__main__':
    summarizer = Summarizer()
    summary = summarizer.summary_refine()