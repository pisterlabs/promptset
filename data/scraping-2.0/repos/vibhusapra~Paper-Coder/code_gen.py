import argparse
import torch
import re
import fitz
import arxiv
import os
import openai 

from tqdm import tqdm
from transformers import LlamaTokenizer 
from langchain.text_splitter import TokenTextSplitter


# Init OpenAI, Text_splitter
openai.api_key = os.environ["OPEN_AI_CODEGEN"]
MODEL_ID = "meta-llama/Llama-2-13b-chat-hf"
tokenizer = LlamaTokenizer.from_pretrained(MODEL_ID)
text_splitter = TokenTextSplitter.from_huggingface_tokenizer(tokenizer=tokenizer,
                                                                            chunk_size = 15000,
                                                                            chunk_overlap = 100,
                                                                            disallowed_special=())

# Scrape Arxiv Papers
def extract_id(url):
    """Extract the paper ID out from the url."""
    doc_id = url.rsplit('/', 1)[-1]
    doc_id = re.match(r"[\d\.]*\d", doc_id)

    return doc_id.group(0) if doc_id else ""

def scrape_arxiv(url):
    """Download and Arxiv paper and return chunked text"""
    url = url.lower()
    paper_id = extract_id(url) # Get paper ID
    search = arxiv.Search(paper_id, max_results=1) # ArXiv API to get paper
    result = next(search.results(), None) 
    title = result.title 
    abstract = result.summary
    doc_file_name = result.download_pdf() # ArXiv API to download paper
    with fitz.open(doc_file_name) as doc_file:
        text = "".join(page.get_text() for page in doc_file)
    os.remove(doc_file_name) # Delete paper
    docs = text_splitter.split_text(text)
    
    return title, docs, abstract

# OpenAI

def generate_completion(prompt, system_prompt, messages = [], max_tokens = 1000, model="gpt-3.5-turbo-16k", temperature=0.0):
    response = openai.ChatCompletion.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            *messages,
            {"role": "user", "content": prompt},
        ],
        temperature=temperature,
        max_tokens=max_tokens,
    )

    return response.choices[0].message.content

# System Prompts
SYSTEM_SUMMARY_PROMPT = """
You are a research assistant that summarizes research papers. 
You provide a high level overview of the paper as well as a specific summary of the experiments ran in the paper.
Do not hallucinate or make up any information not present in the paper."
"""

EXPERIMENT_INSTRUCTION_PROMPT = """
Given the content of the research paper, list down in a clear and linear bullet-point format every step of the experiment(s) described. 
Ensure the steps are detailed enough to allow someone to recreate the experiment and eventually code it.
"""

SYSTEM_ARCHITECTURE_PROMPT = """
"You are a software architect. Given the title, abstract, summary, and experiments of a research paper, 
generate a directory and file structure to recreate the experiments described in the paper. 
The structure should be organized logically and detailed enough to allow someone to recreate the experiment through code."
"""

# Prompt Templates
def summary_template(running_summary, doc):
    template = f"""
    Provide a clear and concise summary of the following chunk of a research paper. 
    Do not hallucinate or make up any information not present in the paper.
    Only repsond with bullet points.
    If applicable, a running summary of previous sections will be provided.

    Running Summary:
    {running_summary}
    Given Paper:
    {doc} 
    """
    return template

def experiment_template(abstract, doc):
    prompt = f"""
    Based on the following research paper section,
    please identify and list down the experimental steps:

    Abstract of paper:
    {abstract}
    Given Paper:
    {doc}
    """
    return prompt

def generate_structure_prompt(title, abstract, summary, experiments):
    prompt = f"""
You are a software architect. Given the title, abstract, summary, and experiments of a research paper, generate a directory and file structure to recreate the experiments described in the paper. The structure should be organized logically and detailed enough to allow someone to recreate the experiment through code.
Only return a directory and file structure. Do not return any explanations.

Example:
Title: Attention Is All You Need
Abstract: We introduce a new, simple network architecture, the Transformer, based solely on attention mechanisms, dispensing with recurrence and convolutions entirely.
Summary: The Transformer, a novel sequence-to-sequence architecture, uses stacked self-attention and point-wise, fully connected layers for both the encoder and decoder. This design shows improvements on various tasks and achieves state-of-the-art results on the English-to-German and English-to-French translation tasks.
Experiments: 
- We trained the Transformer on the WMT 2014 English-to-German and English-to-French datasets.
- Utilized multi-head self-attention mechanism to capture various aspects of the input sequence.
- Used the Adam optimizer with custom learning rate schedules.
- Compared the Transformer's performance with traditional RNN and CNN models.
- Analyzed how varying the number of attention heads impacts performance.
- Introduced label smoothing as a regularization technique.
- Conducted ablation studies to understand the impact of each component of the Transformer.

Directory Structure:
- `project_root/`
    - `data/`
        - `WMT2014/`
            - `English-German/`
                - `train/`
                - `validation/`
                - `test/`
            - `English-French/`
                - `train/`
                - `validation/`
                - `test/`
    - `models/`
        - `Transformer/`
            - `encoder.py`
            - `decoder.py`
            - `attention.py`
            - `config.yaml`
    - `training/`
        - `train.py`
        - `validate.py`
    - `testing/`
        - `test.py`
    - `utils/`
        - `data_preprocessing.py`
        - `label_smoothing.py`
        - `optimizer_schedules.py`
        - `metrics.py`

Given:
Title: {title}
Abstract: {abstract}
Summary: {summary}
Experiments: {experiments}
    """
    return prompt

# Analyze Papers
def process_paper(docs, abstract):
    """Pass chunks of text through OpenAI to summarize the paper."""

    running_summary = []
    experiments = []

    for doc in docs:
        # Summarize the paper
        sum_prompt = summary_template(running_summary=running_summary, doc=doc)
        summary = generate_completion(sum_prompt, system_prompt=SYSTEM_SUMMARY_PROMPT)
        running_summary.append(summary)
        # Save Experiments
        exp_prompt = experiment_template(abstract=abstract, doc=doc)
        experiment_steps = generate_completion(exp_prompt, system_prompt=EXPERIMENT_INSTRUCTION_PROMPT)
        experiments.append(experiment_steps)

    return running_summary, experiments

# Generate Structure
def parse_structure(structure_text):
    lines = structure_text.strip().split("\n")
    paths = []
    
    for line in lines:
        backticks = line.split('`')
        # Check if line has the expected backticks
        if len(backticks) >= 3:
            depth = line.count('-')
            name = backticks[-2]  # Extracts name between backticks
            paths.append((depth, name))
        else:
            # Handle unexpected lines, or simply pass for now
            pass
    
    return paths

def generate_structure(title, abstract, summary, experiments):
    """Generate code structure based on paper."""

    prompt = generate_structure_prompt(title=title, abstract=abstract, summary=summary, experiments=experiments)
    structure = generate_completion(prompt, system_prompt=SYSTEM_ARCHITECTURE_PROMPT)

    return structure

def create_structure(paths):
    stack = []
    created_dirs = set()  # Set to store created directories
    
    for depth, name in paths:
        # Adjust stack for current depth
        while len(stack) > depth:
            stack.pop()
        
        # Create directory or file
        current_path = os.path.join(*stack, name)
        if name.endswith('/'):  # It's a directory
            if current_path not in created_dirs:  # Check if directory is not created already
                os.makedirs(current_path, exist_ok=True)
                created_dirs.add(current_path)
        else:  # It's a file
            open(current_path, 'a').close()  # Create empty file
        
        stack.append(name)

# Removed
# Go through each file and generate the code based on the paper data

# Main function to process a URL
def main():
    parser = argparse.ArgumentParser(description='Summarize Arxiv Paper.')
    parser.add_argument('url', nargs='?', help='Arxiv URL')  # 'nargs=?' makes the URL argument optional
    args = parser.parse_args()

    # If no URL is provided as a command-line argument, ask for one
    if args.url is None:
        args.url = input("Please enter a Arxiv URL: ")
    
    title, docs, abstract = scrape_arxiv(args.url)
    print(f'title: {title}')
    summary, experiments = process_paper(docs=docs, abstract=abstract)
    print('summary: ')
    for i in summary:
        print(i)
    print('')
    print('')
    print('*'*50)
    print('')
    print('')
    print('experiments: ')
    for i in experiments:
        print(i)
    print('')
    print('')
    print('*'*50)
    print('')
    print('')

    # Generate the directory and file structure
    structure_text = generate_structure(title, abstract, summary, experiments)
    print(structure_text)
    paths = parse_structure(structure_text)
    create_structure(paths)

    print(structure_text)
    print(paths)

# Run the main function
if __name__ == "__main__":
    main()
