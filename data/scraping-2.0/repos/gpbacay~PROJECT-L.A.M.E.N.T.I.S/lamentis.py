header = "Local Assistant Model with Enhanced Natural-language-processing and Text-based Intelligent System"
print(header)

#openAI_api_key = "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"
#huggingface_api_key = "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"

import os
import textwrap
import pyttsx3
from dotenv import load_dotenv, find_dotenv
from langchain import HuggingFaceHub
from langchain import PromptTemplate, LLMChain


from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import TextLoader
from langchain.vectorstores import DeepLake

load_dotenv()
os.environ["ACTIVELOOP_TOKEN"] = "eyJhbGciOiJIUzUxMiIsImlhdCI6MTY4NzMxNDI0OCwiZXhwIjoxNzE4OTM2NTc5fQ.eyJpZCI6ImdwYmFjYXkifQ.Yolwz6mFIiWNj7z2R5fvZ-0v6qh4ldNvQY50jUGhkOLlXASPQ8XI_iRSKzrs1nTh2iVLtoxrTmKjGDapQje5kw"
os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY')
source_text = "requirements.txt"
dataset_path = "hub://gpbacay/text_embedding"
documents = TextLoader(source_text).load()
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
docs = text_splitter.split_documents(documents)
db = DeepLake.from_documents(docs, dataset_path=dataset_path, embedding=OpenAIEmbeddings( ) )


def run_lamentis():
    engine = pyttsx3.init()
    voices = engine.getProperty('voices')
    engine.setProperty('voice', voices[1].id)

    def speak(text):
        engine.say(text)
        engine.runAndWait()

    print("Initializing...")
    speak("Initializing LAMENTIS")

    
    load_dotenv(find_dotenv())
    HUGGINGFACEHUB_API_TOKEN = os.environ["HUGGINGFACEHUB_API_TOKEN"]

    
    repo_id = "tiiuae/falcon-7b-instruct"
    falcon_llm = HuggingFaceHub(repo_id=repo_id, model_kwargs={"temperature": 0.1, "max_new_tokens": 500})

    
    template = """Your name is LAMENTIS, a short term for Local Assistant Model with Enhanced Natural-language-processing and Text-based Intelligent System. 
    You are an Artificial Intelligence Personal Virtual Assistant created by Gianne P. Bacay.

    Question: {question}"""

    prompt = PromptTemplate(template=template, input_variables=["question"])
    llm_chain = LLMChain(prompt=prompt, llm=falcon_llm)

    
    response = "System is now online."
    print(response)
    speak(response)
    while True:
        question = input("\ninput: ")
        if "quit" in question:
            break
        response = llm_chain.run(question)
        wrapped_text = textwrap.fill(response, width=200, break_long_words=False, replace_whitespace=False)
        print("LAMENTIS: " + wrapped_text)
        speak(response)

if __name__ == '__main__':
    run_lamentis()


#__________________________python lamentis.py