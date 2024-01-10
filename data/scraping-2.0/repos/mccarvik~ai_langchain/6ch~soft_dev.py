
import sys
sys.path.append("..")
from config import set_environment
set_environment()

from langchain import HuggingFaceHub, HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

def hugface():
    llm = HuggingFaceHub(
        task="text-generation",
        repo_id="HuggingFaceH4/starchat-alpha",
        model_kwargs={
        "temperature": 0.5,
        "max_length": 1000
        }
    )
    text = "a dis"
    print(llm(text))


def small_local():
    checkpoint = "Salesforce/codegen-350M-mono"
    model = AutoModelForCausalLM.from_pretrained(checkpoint)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    pipe = pipeline(
        task="text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=500
    )
    text = """
    def calculate_primes(n):
    \"\"\"Create a list of consecutive integers from 2 up to N.
    For example:
    >>> calculate_primes(20)
    Output: [2, 3, 5, 7, 11, 13, 17, 19]
    \"\"\"
    """
    completion = pipe(text)
    print(completion[0]["generated_text"])
    
    # llm = HuggingFacePipeline(pipeline=pipe)
    # llm(text)


# hugface()
small_local()