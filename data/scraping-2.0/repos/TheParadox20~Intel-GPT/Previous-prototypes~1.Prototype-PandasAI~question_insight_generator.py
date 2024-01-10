import os
import pandas as pd
from langchain.chains.question_answering import load_qa_chain
from langchain.chains import AnalyzeDocumentChain
from langchain.llms import OpenAI

from pandasai import PandasAI

from langchain.llms import OpenAI

from langchain.llms import Ollama
from langchain.llms import AmazonAPIGateway
from langchain.llms import OpenLLM
from langchain.llms import VLLM
from langchain.llms import LlamaCpp
from langchain.llms import VertexAI
from langchain.llms import HuggingFacePipeline

from langchain import PromptTemplate, LLMChain
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

from dotenv import load_dotenv
load_dotenv()

# Load model directly

# !git clone --recursive -j8 https://github.com/abetlen/llama-cpp-python.git
# Make sure the model path is correct for your system!
# llm_1 = LlamaCpp(
#     model_path="C:\Users\ADMIN\Desktop\Projects\Projects\chatcsv\llama-cpp-python\vendor\llama.cpp\models\ggml-vocab-llama.gguf",
#     temperature=0.75,
#     max_tokens=2000,
#     top_p=1,
#     callback_manager=CallbackManager, 
#     verbose=True, # Verbose is required to pass to the callback manager
# )

# llm_1 = VLLM(model="mosaicml/mpt-7b",
#            trust_remote_code=True,  # mandatory for hf models
#            max_new_tokens=128,
#            top_k=10,
#            top_p=0.95,
#            temperature=0.8,
# )

# llm_1 = OpenLLM(
#     model_name="dolly-v2",
#     model_id="databricks/dolly-v2-3b",
#     temperature=0.94,
#     repetition_penalty=1.2,
# )
# api_url = "https://<api_gateway_id>.execute-api.<region>.amazonaws.com/LATEST/HF"
# llm_1 = AmazonAPIGateway(api_url=api_url)

# Initialize the Langchain LLM
# llm_1 = Ollama(base_url="http://localhost:8501/api/generate/", 
#              model="llama2")
# llm_1 = VertexAI()

openai_api_key = os.environ["OPENAI_API_KEY"]
huggingface_api_key = os.environ["HUGGINGFACE_API_KEY"]

def generate_questions_from_csv(input_csv):

    llm = OpenAI(openai_api_key=openai_api_key)
    
    # Check if the input_csv is None or empty
    if input_csv is None:
        return "Error: CSV file is missing or empty."

    try:
        # Load the CSV file into a DataFrame
        df = pd.read_csv(input_csv)

        # Check if the DataFrame is empty
        if df.empty:
            return "Error: CSV file is empty."

        # Create a question-answering chain
        qa_chain = load_qa_chain(llm, chain_type="map_reduce")

        # Create an AnalyzeDocumentChain
        qa_document_chain = AnalyzeDocumentChain(combine_docs_chain=qa_chain)

        # Define a question to generate questions for plotting relevant visualizations
        question = "Write 2 statement that involves plotting relevant visualizations that would drawing insights from this data with the best type of visualization to use and add the word 'plot'."

        # Generate questions using Langchain
        output_document = qa_document_chain.run(input_document=df.to_string(index=False), question=question)

        return output_document
    except Exception as e:
        return f"Error: {str(e)}"
# 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
print(generate_questions_from_csv('HappyCountries.csv'))