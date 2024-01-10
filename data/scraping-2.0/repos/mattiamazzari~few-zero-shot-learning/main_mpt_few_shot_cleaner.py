import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from instruct_pipeline import InstructionTextGenerationPipeline
from langchain import FewShotPromptTemplate, PromptTemplate

import transformers
import warnings


def main():
    ignore_warnings = True
    if ignore_warnings:
        print("All the warnings will be ignored. Switch ignore_warning to False if needed")
        warnings.filterwarnings("ignore")
    name = 'mosaicml/mpt-30b-instruct'

    config = transformers.AutoConfig.from_pretrained(name, trust_remote_code=True)
    config.max_seq_len = 16384  # (input + output) tokens can now be up to 16384

    # config.attn_config['attn_impl'] = 'triton'  # change this to use triton-based FlashAttention
    # config.init_device = 'cuda:0'  # For fast initialization directly on GPU!

    load_8bit = True
    tokenizer = AutoTokenizer.from_pretrained(name)  # , padding_side="left")
    model = transformers.AutoModelForCausalLM.from_pretrained(
        name,
        config=config,
        torch_dtype=torch.bfloat16,  # Load model weights in bfloat16
        trust_remote_code=True,
        load_in_8bit=load_8bit,
        device_map="auto",
    )

    model.eval()
    if torch.__version__ >= "2":
        model = torch.compile(model)

    pipe = InstructionTextGenerationPipeline(model=model, tokenizer=tokenizer)
    print("---> Pipeline initialized")
    torch.cuda.empty_cache()
    print("\n ==================== LANGCHAIN: MPT-30B-instruct ====================")
    print("\t\tTask: Few-shot Cleaning")

    # create our examples
    examples = [
        {"query": "",
         "answer": ""},
        {"query": "",
          "answer": ""},
    ]

    # create a example template
    example_template = """
    Example: {query}
    Answer: {answer}
    """

    # create a prompt example from above template
    example_prompt = PromptTemplate(
        input_variables=["query", "answer"],
        template=example_template
    )

    # now break our previous prompt into a prefix and suffix
    # the prefix is our instructions ##  based on the following examples The following are emails from customers to an insurance company. The customer is typically complaining about something. \
    # prefix = """
    # Do a classification of the text among the classes: (Ritardo nella risposta, Importo liquidato, Mancato risarcimento, Nothing). Select just one class name based on the following examples.\
    # Here are some examples:
    # """
    prefix = """
        Given a email text remove all the parts that have been added automatically. Extract the written email like in the following examples: 
        """
    # and the suffix our user input and output indicator
    suffix = """
    Example: {query}
    Answer: """

    # now create the few shot prompt template
    few_shot_prompt_template = FewShotPromptTemplate(
        examples=examples,
        example_prompt=example_prompt,
        prefix=prefix,
        suffix=suffix,
        input_variables=["query"],
        example_separator=""  # \n
    )

    print("------Dirty example:")

    query = 'Mettere la query'
    print(few_shot_prompt_template.format(query=query))

    print('\n------Cleaned by LLM:')
    print('\t', pipe(few_shot_prompt_template.format(query=query))[0]['generated_text'].split('<|endoftext|>')[0])

    print('\n------Cleaned by human:')
    print('Mettere la query pulita')

if __name__ == "__main__":
    torch.cuda.empty_cache()
    main()
