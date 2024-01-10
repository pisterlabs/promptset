"""
This script contains helper functions for handling the
splitting and picking the random chunks of text to generate
questions from. There is also a function for using an LLM
to create an evaluation dataset for us.

"""
import os
import random
import datetime
import pandas as pd

from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import QAGenerationChain
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

def _pick_random_chunks(num_qa_pairs: int, max_index: int) -> list:
    """Pick N random sections of split documents to generate questions from."""
    return random.sample(range(0, max_index + 1), num_qa_pairs)

def _split_document_into_chunks(doc_file: str, chunk_size: int = 3000, chunk_overlap: int = 250) -> list:
    """Split the document into equal-sized chunks."""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, 
                                                   chunk_overlap=chunk_overlap, 
                                                   length_function=len)
    loader = PyPDFLoader(doc_file)
    return loader.load_and_split(text_splitter=text_splitter)

def _initialize_qa_chain() -> QAGenerationChain:
    """Generate QA evaluation dataset"""

    template = """You are a smart assistant designed to come up with meaninful question and answer pair. The question should be to the point and the answer should be as detailed as possible.
    Given a piece of text, you must come up with a question and answer pair that can be used to evaluate a QA bot. Do not make up stuff. Stick to the text to come up with the question and answer pair.
    When coming up with this question/answer pair, you must respond in the following format:
    ```
    {{
        "question": "$YOUR_QUESTION_HERE",
        "answer": "$THE_ANSWER_HERE"
    }}
    ```

    Everything between the ``` must be valid json.

    Please come up with a question/answer pair, in the specified JSON format, for the following text:
    ----------------
    {text}
    """

    prompt = PromptTemplate.from_template(template)
    llm = ChatOpenAI(temperature=0.9)
    return QAGenerationChain.from_llm(llm=llm, prompt=prompt)

def generate_eval_dataset(doc_file, chunk_size=1000, num_qa_pairs = 10,  save_to_file: bool = False):
    """ 
    Generate evaluation dataset: This will break up the document into
    equal size chunks in characters based on the chunk_size.
    If it's small, like chunk_size=100, then a question will be made for a single sentence, 
    if chunk_size=1000 then a single question will be generated for a paragraph
    NOTE: this means to more questions you want for evaluation
    the smaller the source information to generate that question will be, 
    this will lead to a point of deminishing returns as there won't be enough
    contextual information if chunk_size is too small, and may miss out on potential
    evaluations if the chunk_size is too big.
    """
    
    qa_chain = _initialize_qa_chain()
    split_docs = _split_document_into_chunks(doc_file, chunk_size=chunk_size, chunk_overlap=5)
    
    len_split_docs = len(split_docs)
    if num_qa_pairs > len_split_docs:
        raise ValueError("You've requested more questions than number of splits in the document"
                         "Consider reducing the size of your chunk_size.")
    print(f"Number of split docs: {len_split_docs}")

    # Generate questions based on random sections of the documents
    # max_index represents the range from which to generate random indices
    random_chunks = _pick_random_chunks(num_qa_pairs=num_qa_pairs, max_index=len(split_docs))
    print(f"Random Chunks: {random_chunks}")
    
    # Generate evaluation dataset
    qa_pairs = [item for idx, doc in enumerate(split_docs) if idx in random_chunks for item in qa_chain.run(doc.page_content)]

    # Convert the list of dictionaries into a DataFrame
    qa_df = pd.DataFrame(qa_pairs)
    print(qa_df.head(5))

    if save_to_file:
        if not os.path.exists("./eval_data/"):
            os.makedirs("./eval_data/")

        current_time = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"./eval_data/gen_qa_{current_time}.csv"
        qa_df.to_csv(filename, index=False)
        print(f"Saved QA pairs to {filename}")
        
    return qa_pairs