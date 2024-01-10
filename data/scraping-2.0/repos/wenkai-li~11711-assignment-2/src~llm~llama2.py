from tqdm import tqdm
from langchain.llms import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForCausalLM
from langchain.chains import LLMChain, SimpleSequentialChain
from langchain.prompts import PromptTemplate
from fuzzywuzzy import fuzz
from fuzzywuzzy import process
import transformers
import torch
import langchain
from langchain.cache import InMemoryCache
import re, gc, csv
import numpy as np
import spacy

def llama2(system_prompt, sentences, output_path):
    langchain.llm_cache = InMemoryCache()
    model_id = "/data/datasets/models/huggingface/meta-llama/Llama-2-70b-chat-hf"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto",load_in_8bit=True)
    pipeline = transformers.pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_length=4096,
        temperature=1,
        top_k=1,
        eos_token_id=tokenizer.eos_token_id,
    )
    llm = HuggingFacePipeline(pipeline=pipeline)
    msg_1 = "Are you clear about your role?"
    answer_1 = "Sure, I'm ready to help you with your NER task. Please provide me with the necessary information to get started."
    prompt_template1 = """<s>[INST] <<SYS>>
{system_prompt}
<</SYS>>

{user_msg} [/INST] {model_answer} </s><s>[INST]  Entity Definition:
1. MethodName: Names of the baseline and proposed systems, could be: the main description of the method, such as: "Bidirectional Encoder Representations from Transformers" or the abbreviated form of their method, such as: "BERT".
2. HyperparameterName: Names of the hyper-parameters mentioned in the paper, that cannot be inferred while fitting models to the training data. This could either be the full description (e.g., "number of layers"), or the mathematical notation. "train/dev/test split ratio" should be labeled as HyperparameterName.
3. HyperparameterValue: Value of the hyper-parameters. All hyperparameter values annotated should be numerical values.
4. MetricName: Names of the evaluation metrics being used for method evaluation. Only annotate the name of the metric and not include other context. For example, given a string "the accuracy on test set" you should only annotate "accuracy". The abbreviations are also considered valid metric names.
5. MetricValue: Evaluation results of methods on each metric. Many analyses use relative metric values (+5.3%) instead of absolute values (45.6%), these relative values should also be annotated as valid metric values. Be sure to include "%" at the end of the number.
6. TaskName: Name of the tasks that the current work is evaluated on, e.g. "Named Entity Recognition". You should not annotate tasks that are mentioned but not evaluated with the proposed work, or the names that do not provide information about what task is being solved, e.g. "task A", "subtask A" should not be annotated.
7. DatasetName: Name of the dataset of target tasks. Some works evaluate on dataset benchmarks (i.e., a collection of datasets) such as GLUE. You could also label the benchmark name as a dataset name.

Output Format:
{{'MethodName': [list of entities present], 'HyperparameterName': [list of entities present], 'HyperparameterValue': [list of entities present], 'MetricName': [list of entities present], 'MetricValue': [list of entities present], 'TaskName': [list of entities present], 'DatasetName': [list of entities present]}}
If no entities are presented in any categories keep it None

Examples 1: Sentence: "We denote the number of layers as L , the hidden size as H , and learning rate as θ ( θ is set to 0.001 ) ."
Output: {{'MethodName': None, 'HyperparameterName': ['number of layers', 'L', 'hidden size', 'H', 'learning rate', 'θ'], 'HyperparameterValue': ['0.001'], 'MetricName': None, 'MetricValue': None, 'TaskName': None, 'DatasetName': None}}

Examples 2: Sentence: "Spearman correlations are reported for STS - B, and accuracy scores are reported for the other tasks ."
Output: {{'MethodName': None, 'HyperparameterName': None, 'HyperparameterValue': None, 'MetricName': ['spearman correlations', 'accuracy'], 'MetricValue': None, 'TaskName': None, 'DatasetName': 'STS-B'}}

Examples 3: Sentence: "BERT outperforms other methods on natural language inference ( NLI ) , question answering . It was not evaluated on machine translation."
Output: {{'MethodName': ['BERT'], 'HyperparameterName': None, 'HyperparameterValue': None, 'MetricName': None, 'MetricValue': None, 'TaskName': ['natural language inference', 'NLI', 'question answering'], 'DatasetName': None}}

4. Sentence: "{sentence}"
Output: [/INST]"""

    prompt1 = PromptTemplate(template=prompt_template1, input_variables=['sentence'], partial_variables={"system_prompt": system_prompt, "user_msg": msg_1, "model_answer": answer_1})
    # prompt_template2 = "[INST] <<SYS>>\n{input}\n{format_instructions}\n<</SYS>>[/INST]"
    # prompt2 = PromptTemplate(template=prompt_template2, input_variables=['input'], partial_variables={"format_instructions": parser.get_format_instructions()})
    generate_chain = LLMChain(llm=llm, prompt=prompt1)
    # json_chain = LLMChain(llm=llm, prompt=prompt2)
    # overall_chain = SimpleSequentialChain(chains=[generate_chain, json_chain], verbose=True)
    # retry_parser = RetryWithErrorOutputParser.from_llm(
    #             parser=parser, llm=llm)
    # autofix_parser = OutputFixingParser.from_llm(parser=parser, llm=llm)
    # chain = LLMChain(llm=llm, prompt=prompt)
    # outputs = []
    fd = open(output_path,'w')
    for sentence in tqdm(sentences):
        output = generate_chain.run(sentence)
        # outputs.append(output)
        dict = extract_information(output)
        format_dict = create_formatted_dictionary(dict)
        conll = convert_to_conll_format(sentence,format_dict)
        fd.write(conll)
    # return outputs

def fuzzy_match_keys(input_key, possible_keys):
    # Find the closest matching key from possible_keys using fuzzy matching
    best_match, score = process.extractOne(input_key, possible_keys)
    if score >= 50:  # You can adjust the threshold for a better match
        return best_match
    else:
        return None

def create_formatted_dictionary(input_dict):
    # Define possible keys
    possible_keys = [
        'MethodName', 'HyperparameterName', 'HyperparameterValue',
        'MetricName', 'MetricValue', 'TaskName', 'DatasetName'
    ]

    # Initialize the formatted dictionary with None values
    formatted_dict = {key: None for key in possible_keys}

    for input_key, value in input_dict.items():
        matched_key = fuzzy_match_keys(input_key, possible_keys)
        if matched_key:
            formatted_dict[matched_key] = value

    return formatted_dict

def text_to_sentences(text):
    # Process the text using spaCy
    doc = nlp(text)

    # Extract the sentences from the processed document
    sentences = [sent.text for sent in doc.sents]

    return sentences

def convert_to_conll_format(sentence, data):
    words = sentence.split()
    labels = ['O'] * len(words)

    for entity_type, entity_values in data.items():
        if entity_values is not None:
            for entity_value in entity_values:
                try: 
                    entity_tokens = entity_value.split()
                    entity_start = 'B-' + entity_type
                    entity_inside = 'I-' + entity_type

                    for i in range(len(words) - len(entity_tokens) + 1):
                        if words[i:i+len(entity_tokens)] == entity_tokens:
                            labels[i] = entity_start
                            for j in range(i + 1, i + len(entity_tokens)):
                                labels[j] = entity_inside
                except:
                    continue

    conll_lines = []
    for word, label in zip(words, labels):
        conll_lines.append(f"{word} {label}")

    return '\n'.join(conll_lines) + '\n'

def extract_information(text):
    # Extract the content inside the curly braces
    content_match = re.search(r'\{([^}]*)\}', text)
    if content_match:
        content = content_match.group(1)
        try:
            dict = eval('{'+content+'}')
            return dict
        except:
            print("parse fail"+content)
    # Return a dictionary with all values set to None if the specified format is not found
    return {
        'MethodName': None,
        'HyperparameterName': None,
        'HyperparameterValue': None,
        'MetricName': None,
        'MetricValue': None,
        'TaskName': None,
        'DatasetName': None
    }

if __name__ == "__main__":
    torch.cuda.empty_cache()
    gc.collect()
    # Load the spaCy model with the sentencizer
    nlp = spacy.load("en_core_web_sm")
    output_file = 'predict.txt'
    system_prompt_path = 'system_prompt.txt'
    system_prompt = open(system_prompt_path,encoding='utf-8',mode='r').read()
    outputs = []
    input_path_conll = 'test.conll'
    batch = np.loadtxt(input_path_conll,dtype=str,delimiter=' ')

    # Combine the list of tokens into a single string
    text = " ".join(batch[...,0])
    sentences = text_to_sentences(text)
    # print(sentences)
    llama2(system_prompt, sentences, output_file)
        
    
