import json
import os
import torch
from transformers import pipeline
from langchain import PromptTemplate, LLMChain
from langchain.llms import HuggingFacePipeline
from langchain.prompts.example_selector import SemanticSimilarityExampleSelector
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.prompts import FewShotPromptTemplate
import argparse


def get_data_from_rebel_files(rebel_path, file_name):
    data = []
    with open(os.path.join(rebel_path, file_name), 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    return data

def get_examples(data):
    examples = []
    #for instance in data[:5]:
    for instance in data:
        input_text = instance['input']
        target_triples = str(instance['meta_obj']['substring_triples'])
        examples.append({"input_text": input_text, "target_triples": target_triples})
    return examples


def main():
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--rebel_path", type=str, default="/home/finapolat/GenIE/data/rebel")
    parser.add_argument("--extraction_path", type=str, default="/home/finapolat/KGC-LLM/extractions")
    parser.add_argument("--few_shot_amount", type=int, default=3)
    args = parser.parse_args()
    
    example_prompt = PromptTemplate(
                                    input_variables=["input_text", "target_triples"],
                                    template="""A triple has three components: subject, relations, object. Extract triples from the given text in the following format: (subject, relation, object). 
                                    Text to extract triples: {input_text} Extracted Triples: {target_triples}"""
                                    )
    examples = get_examples(get_data_from_rebel_files(args.rebel_path, "en_val_small.jsonl"))
    
    example_selector = SemanticSimilarityExampleSelector.from_examples(
                        # This is the list of examples available to select from.
                        examples, 
                        # This is the embedding class used to produce embeddings which are used to measure semantic similarity.
                        #HuggingFaceInstructEmbeddings(), 
                        HuggingFaceEmbeddings(),
                        # This is the VectorStore class that is used to store the embeddings and do a similarity search over.
                        Chroma, 
                        # This is the number of examples to produce.
                        k=args.few_shot_amount,
                        )
    # Finally, we create the `FewShotPromptTemplate` object.
    few_shot_prompt = FewShotPromptTemplate(
                        # These are the examples we want to insert into the prompt.
                        example_selector=example_selector,
                        # This is how we want to format the examples when we insert them into the prompt.
                        example_prompt=example_prompt,
                        # The prefix is some text that goes before the examples in the prompt.
                        # Usually, this consists of intructions.
                        prefix="""A triple has three components: subject, relations, object. Extract triples from the given text in the following format: ['subject', 'relation', 'object'] and put them in a list. Here are some examples: """,
                        # The suffix is some text that goes after the examples in the prompt.
                        # Usually, this is where the user input will go
                        suffix="""End of the examples. \nText to extract triples: {input} \n Extracted Triples: """,
                        # The input variables are the variables that the overall prompt expects.
                        input_variables=["input"],
                        # The example_separator is the string we will use to join the prefix, examples, and suffix together with.
                        example_separator="\n",
                        )
    
    print("Loading model...")
    generate_text = pipeline(model="databricks/dolly-v2-3b", torch_dtype=torch.bfloat16,
                            trust_remote_code=True, device_map="auto", return_full_text=True)
    #generate_text = pipeline(model="google/flan-t5-large", 
       #                  torch_dtype=torch.bfloat16, trust_remote_code=True,  
        #                 device_map="auto")
                         
    hf_pipeline = HuggingFacePipeline(pipeline=generate_text)
    
    llm_fewshot_chain = LLMChain(llm=hf_pipeline, prompt=few_shot_prompt)
    print("Model loaded.")
    
    print("Loading test data...")
    test_data = get_data_from_rebel_files(args.rebel_path, "en_test.jsonl")
    print("Test data loaded.")
    print(f"Number of test instances: {len(test_data)}")
    outfile = open(os.path.join(args.extraction_path, "fewshot_extraction_dolly_on_REBEL.txt"), "w", encoding="utf-8")
    
    print("Starting few-shot extraction...")
    #for d in test_data[:5]:
    for d in test_data:
        input_text = d["input"]
        extraction = llm_fewshot_chain.run(input=input_text)
        extraction_dict = {"model_name": "google/flan-t5-large",
                           "dataset": "REBEL_en_test",
                           "input_text": input_text, 
                           "few-shot extraction": extraction}
        #print(extraction_dict)
        outfile.write(json.dumps(extraction_dict) + "\n")
    outfile.close()
    print("Few-shot extraction finished.")
    print("Extractions are written to outfile.")
    
if __name__ == "__main__":
    main()