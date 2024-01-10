from langchain.llms import LlamaCpp
from langchain.chat_models import ChatOpenAI
from langchain.chains.llm import LLMChain
from langchain.prompts import PromptTemplate
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.cache import SQLiteCache
import langchain

import itertools

from utils import setup_logger

from dotenv import load_dotenv

import os

# Load the .env file
load_dotenv()

OPEN_AI_KEY = os.getenv("OPEN_AI_KEY")

os.environ["OPENAI_API_KEY"] = OPEN_AI_KEY

logger = setup_logger('contr_detector_logger', 'app.log')
langchain.llm_cache = SQLiteCache(database_path=".langchain.db")


callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
llm_llama = LlamaCpp(
        # model_path="llama-2-7b.Q4_K_M.gguf",
        model_path="models/OpenOrca-Platypus2-13B-Q4_K_M.gguf",
        temperature=0,
        max_tokens=1000,
        top_p=3,
        callback_manager=callback_manager,
        verbose=True,  # Verbose is required to pass to the callback manager
    )

# TODO: move the prompt to a file to be configured
prompt_template = """
            Statement 1: {doc1}
            Statement 2: {doc2}

            Question: Are these two statements contradictory? Answer "yes" or "no".
        """

prompt = PromptTemplate.from_template(prompt_template)

llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo", openai_api_key=OPEN_AI_KEY)
llm_chain = LLMChain(llm=llm, prompt=prompt)

def detect_contradictions(documents, metadatas, model_type: str):
    contrs = []
    for doc1, doc2 in itertools.combinations(zip(documents, metadatas), 2):
        # print(doc1)

        doc1, meta1 = doc1
        doc2, meta2 = doc2

        if model_type == "openAI":
            llm = llm_chain
            result = llm_chain({"doc1": doc1, "doc2": doc2}, return_only_outputs=True)
            print(result)
            if "yes" in result['text'].lower():
                logger.info(f"Contradiction: {doc1} {doc2}")
                print(f"Contradiction: {doc1} {doc2}")
                contrs.append(((doc1, meta1), (doc2, meta2)))
                # break # TODO: remove
            else:
                logger.info(f"No contradiction: {doc1} {doc2}")
                print(f"No contradiction: {doc1} {doc2}")
        else: 
            llm = llm_llama

            prompt = f"""
                Statement 1: {doc1}
                Statement 2: {doc2}

                Question: Are these two statements contradictory? Answer "yes" or "no".
            """

            if "yes" in llm(prompt).lower():
                logger.info(f"Contradiction: {doc1} {doc2}")
                print(f"Contradiction: {doc1} {doc2}")
                contrs.append(((doc1, meta1), (doc2, meta2)))
            else:
                logger.info(f"No contradiction: {doc1} {doc2}")
                print(f"No contradiction: {doc1} {doc2}")
    
    print("Done with checking for contradictions")
    print(contrs)
    return contrs