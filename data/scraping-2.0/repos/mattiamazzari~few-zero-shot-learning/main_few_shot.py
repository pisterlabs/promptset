import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from instruct_pipeline import InstructionTextGenerationPipeline
from langchain import FewShotPromptTemplate, PromptTemplate

import transformers
import warnings

import pandas as pd
import numpy as np


def main():

    # path_examples = 'E:\\llm\\data\\210723_train_subset_3.xlsx'
    # path_subset = 'E:\\llm\\data\\210723_valid_set.xlsx'

    # df_examples = pd.read_excel(path_examples)
    # df_subset = pd.read_excel(path_examples)


    ignore_warnings = True
    if ignore_warnings:
        print("All the warnings will be ignored. Switch ignore_warning to False if needed")
        warnings.filterwarnings("ignore")
    name = 'mosaicml/mpt-30b-instruct'

    config = transformers.AutoConfig.from_pretrained(name, trust_remote_code=True)
    # config.max_seq_len = 16384 # 83968
    load_8bit = True
    if not load_8bit:
        load_4bit = True
    else:
        load_4bit = False

    # from transformers import BitsAndBytesConfig
    #
    # nf4_config = BitsAndBytesConfig(
    #     load_in_4bit=True,
    #     bnb_4bit_quant_type="nf4",
    #     bnb_4bit_use_double_quant=True,
    #     bnb_4bit_compute_dtype=torch.bfloat16
    # )

    tokenizer = AutoTokenizer.from_pretrained(name)  # , padding_side="left")
    model = transformers.AutoModelForCausalLM.from_pretrained(
        name,
        config=config,
        torch_dtype=torch.bfloat16,  # Load model weights in bfloat16
        trust_remote_code=True,
        load_in_8bit=load_8bit,
        # load_in_4bit=load_4bit,
        #quantization_config=nf4_config,
        device_map="auto",
    )

    model.eval()
    if torch.__version__ >= "2":
        model = torch.compile(model)

    print("--PIPELINE INIT--")
    pipe = InstructionTextGenerationPipeline(model=model, tokenizer=tokenizer)

    torch.cuda.empty_cache()
    print("\n ==================== LANGCHAIN: MPT-30B-instruct ====================")
    print("\t\tTask: Few-shot Prompting")

    print("\nLoading examples")
    examples = []
    list_classes = []
    # create our examples
    c = 0

    for i, row in df_examples.iterrows():
        c+=1
        if c%2 ==0:
            continue
        else:
            examples.append({"query": row['Sintesi_cleaned'], "answer":row['Domanda']})
            if row['Domanda'] not in list_classes:
                list_classes.append(row['Domanda'])

        # create our examples

    # create a example template
    example_template = """
    Text: {query}
    Class: {answer}
    """

    # create a prompt example from above template
    example_prompt = PromptTemplate(
        input_variables=["query", "answer"],
        template=example_template
    )

    # now break our previous prompt into a prefix and suffix
    # the prefix is our instructions ##  based on the following examples The following are emails from customers to an insurance company. The customer is typically complaining about something. \
    prefix = f"""
    Do a classification of the text among the classes: {list_classes}. Select just one class name based on the following examples.\
    Here are some examples:
    """
    # prefix = f"""
    #     Classifica il seguente testo tra queste classi : {list_classes}. Seleziona una sola classe basandoti sugli esempi seguenti.\
    #     Qua gli esempi:
    #     """
    # and the suffix our user input and output indicator
    suffix = """
    User: {query}
    AI: """

    # now create the few shot prompt template
    few_shot_prompt_template = FewShotPromptTemplate(
        examples=examples,
        example_prompt=example_prompt,
        prefix=prefix,
        suffix=suffix,
        input_variables=["query"],
        example_separator="" # \n
    )

    print("------Template:")
    query = df_subset.iloc[0]['Sintesi_cleaned']
    # print('\t', pipe(few_shot_prompt_template.format(query=query))[0]['generated_text'].split('<|endoftext|>')[0])

    # print(few_shot_prompt_template.format(query=query))
    first = True
    count = 0
    torch.cuda.empty_cache()
    for i, row in df_subset.iterrows():
        query = row['Sintesi_cleaned']
        ground_truth = row['Domanda']
        if first:
            print(f'\n------Risposta {i}:')
            print('\t', pipe(few_shot_prompt_template.format(query=query))[0]['generated_text'].split('<|endoftext|>')[0])
            print(f'\n------Ground truth {i}: {ground_truth}')
        else:
            print(f'\n------Reclamo {i}:')
            print('\t', query)
            print('\n------Risposta {i}:')
            print('\t', pipe(few_shot_prompt_template.format(query=query))[0]['generated_text'].split('<|endoftext|>')[0])
            print(f'\n------Ground truth {i}: {ground_truth}')
        torch.cuda.empty_cache()
        count +=1
        if count == 20:
            break
if __name__ == "__main__":
    torch.cuda.empty_cache()
    main()
