import argparse
import os
import sys
sys.path.append("../.")

from config import Config
from langchain.prompts.few_shot import FewShotPromptTemplate
from langchain.prompts.prompt import PromptTemplate
from langchain.prompts.example_selector import SemanticSimilarityExampleSelector
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings


from util_functions import get_examples_for_prompt



def get_prompt(args):
    """
    Get few-shot prompt
    """

    example_selector = SemanticSimilarityExampleSelector.from_examples(
        # This is the list of examples available to select from.
        args.examples,
        # This is the embedding class used to produce embeddings which are used to measure semantic similarity.
        OpenAIEmbeddings(),
        # embeddings = OpenAIEmbeddings(
        #         deployment="text-embedding-ada-002",
        #         model="text-embedding-ada-002",
        #         openai_api_base="https://hlt-nlp.openai.azure.com/",
        #         openai_api_type="azure",
        #     ),
        # OpenAIEmbeddings(openai_api_key=""),
        # This is the VectorStore class that is used to store the embeddings and do a similarity search over.
        vectorstore_cls = Chroma,
        # This is the number of examples to produce.
        k=2
        )


    prompt = FewShotPromptTemplate(
        example_selector=example_selector, 
        # examples = args.examples[:args.num_examples],
        example_prompt=args.example_prompt, 
        prefix = args.prompt_config.prompt_prefix,
        suffix = args.prompt_config.prompt_suffix, 
        input_variables=["Text"]
    )

    return prompt

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--lang', '-l', default='it')
    args = parser.parse_args()
    assert args.lang in ['it', 'es', 'eu'], "The language must be one of 'it', 'es', 'eu'"

    configs = Config(args.lang)
    args.config = configs

    args.prompt_doc_ids = few_shot_doc_ids



    args.examples = get_examples_for_prompt(args)
    args.new_clinical_stmt = sample_it_clinical_stmt

    get_prompt(args)