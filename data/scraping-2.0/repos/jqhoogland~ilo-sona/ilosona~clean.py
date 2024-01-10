"""
Tooling to clean up the corpus (using GPT-4 via the OpenAI API).
"""
import os

from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAI
from langchain.output_parsers import PydanticOutputParser
from langchain.prompts import (ChatPromptTemplate, HumanMessagePromptTemplate,
                               PromptTemplate)
from langchain.text_splitter import CharacterTextSplitter
from pydantic import BaseModel, Field, validator
from tqdm import tqdm
from transformers import GPT2TokenizerFast


def process_file(path, text_splitter, prompt, parser, model):
    with open(path, 'r') as f:
        text = f.read()
    texts = text_splitter.split_text(text)
    
    cleaned_data = []
    
    for text in tqdm(texts, desc=f"Processing {path}"):
        formatted_prompt = prompt.format(fragment=text)
        response = model(formatted_prompt)  # Assuming model is callable with formatted_prompt as input
        parsed_response = parser.parse(response)
        cleaned_data.append(parsed_response)
    
    cleaned_file_path = path.replace("./corpus", "./corpus-cleaned")
    os.makedirs(os.path.dirname(cleaned_file_path), exist_ok=True)
    with open(cleaned_file_path, 'w') as f:
        for item in cleaned_data:
            f.write(str(item))
        print(f"Saved cleaned data to {cleaned_file_path}")


def process_directory(dir_path, text_splitter, prompt, parser, model):
    for root, dirs, files in tqdm(os.walk(dir_path), "Processing files..."):
        for file in tqdm(files, desc=f"Processing {root}"):
            if file.endswith('.txt'):  # Assuming we're processing text files
                file_path = os.path.join(root, file)
                process_file(file_path, text_splitter, prompt, parser, model)


# Define your desired data structure.
class TokiPonaSample(BaseModel):
    input: str = Field(description="sample text to clean")
    output: str = Field(description="cleaned sample text")


def clean(dir_path, chunk_size=1024, model_name='text-davinci-003', temperature=0.0):
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    text_splitter = CharacterTextSplitter.from_huggingface_tokenizer(tokenizer, chunk_size=chunk_size, chunk_overlap=0)
 
    model = OpenAI(model_name=model_name, temperature=temperature)

    parser = PydanticOutputParser(pydantic_object=TokiPonaSample)
    prompt = PromptTemplate(
        template="Clean up the following fragment of text in toki pona. Remove long strings of other languages, "
                 "except when used to clarify a name like 'jan Lejonato (Leonardo)'. Dates, urls, etc. are fine.\n"
                 "{format_instructions}\n{fragment}",
        input_variables=["fragment"],
        partial_variables={"format_instructions": parser.get_format_instructions()}
    )

    process_directory(dir_path, text_splitter, prompt, parser, model)


if __name__ == '__main__':
    clean("./corpus-test")
