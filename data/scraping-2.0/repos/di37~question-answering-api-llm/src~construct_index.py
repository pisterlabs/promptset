import os, sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))

from gpt_index import (
    SimpleDirectoryReader,
    GPTSimpleVectorIndex,
    LLMPredictor,
    PromptHelper,
)
from langchain import OpenAI


from dotenv import load_dotenv

load_dotenv()

# Setting the API KEY which can be found from OpenAI once signed up.
# For this project, the API KEY is stored in .env file.
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

from utils import *


def ConstructIndex():
    """
    Constructs a vector index file after passing the path of data via langchain and gpt-index libraries.
    This procedure is known as fine-tuning or training embedding.
    """
    # set maximum input size
    max_input_size = 4096*2
    # set number of output tokens
    num_outputs = 256*2
    # set maximum chunk overlap
    max_chunk_overlap = 20*2
    # set chunk size limit
    chunk_size_limit = 600*2
    # The above variables can be changed as per the project requirements.

    # If the index file does not exist, then create it.
    if not os.path.exists(INDEX_FILE_PATH):
        # PromptHelper helps in designing the prompts and taking the prompts
        # based on input text
        prompt_helper = PromptHelper(
            max_input_size,
            num_outputs,
            max_chunk_overlap,
            chunk_size_limit=chunk_size_limit,
        )

        # define Large Language Model. We can also use Hugging Face as an alternative.
        llm_predictor = LLMPredictor(
            # For more variability in our answers, we can differ the temperature.
            # For better performance, we can choose text-davinci-003 model.
            llm=OpenAI(temperature=0, model_name="text-davinci-003", max_tokens=num_outputs)
        )

        # Loading the data from .txt file into documents for gpt_index library
        # to create index file based on it. In other words, it is embedding file from which
        # OpenAI model learnt from .txt file.
        documents = SimpleDirectoryReader(DATA_PATH).load_data()

        # construct vector index file via means of gpt-index library
        index = GPTSimpleVectorIndex(
            documents, llm_predictor=llm_predictor, prompt_helper=prompt_helper
        )
        # save index to disk
        index.save_to_disk(INDEX_FILE_PATH)
        # Once index file is saved to the disk, we won't need to call OpenAI library
        # again to define LLM. Embeddings is done in such a way that when we will
        # use gpt-index library, we will be querying it only and based on the
        # query, we will be able to get answers that we want.
        return {"Response": "Index file created successfully."}
    else:
        return {"Response": "Index file already existed."}


# if __name__ == "__main__":
#     index = ConstructIndex()
#     print(index)
