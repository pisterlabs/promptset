import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline

import openai
from langchain.chains.summarize import load_summarize_chain
from langchain.chat_models import ChatOpenAI  # for generating the summary
from langchain.docstore.document import Document  # for storing the text
from langchain.prompts import PromptTemplate  # Template for the prompt



def summarize_text_with_hugging_face(
    text, model_name="facebook/bart-large-cnn", max_length=250, min_length=25
):
    """
    Summarize the given text using the specified model.

    Parameters:
    - text (str): The text to summarize.
    - model_name (str): The name of the pre-trained model to use.
    - max_length (int): The maximum length of the summary.
    - min_length (int): The minimum length of the summary.

    Returns:
    - str: The summary of the text.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    summarization_pipeline = pipeline(
        "summarization", model=model_name, tokenizer=tokenizer
    )  # device=0 if you have a GPU, if not specified will be run on CPU by default
    summary = summarization_pipeline(
        text, max_length=max_length, min_length=min_length, do_sample=False
    )
    return summary[0]["summary_text"]


def summarize_text_with_open_ai(blog_text, prompt_template):
    model_name = "gpt-3.5-turbo"

    # Converts each part into a Document object
    docs = [Document(page_content=blog_text)]

    # Loads the lanugage model
    llm = ChatOpenAI(temperature=0, openai_api_key=openai.api_key, model_name=model_name)

    # Defines prompt template
    prompt = PromptTemplate(template=prompt_template, input_variables=["text"])

    # Define model parameters
    verbose = False  # If set to True, prints entire un-summarized text

    # Loads appropriate chain based on the number of tokens. Stuff or Map Reduce is chosen
    chain = load_summarize_chain(
        llm,
        chain_type="stuff",
        prompt=prompt,
        verbose=verbose,
    )
    summary = chain.run(docs)

    return summary
