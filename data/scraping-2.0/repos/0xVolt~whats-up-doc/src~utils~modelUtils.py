import inquirer
from .fileUtils import returnModelLocalPath
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chains import LLMChain
from langchain.llms import GPT4All
from langchain.prompts import PromptTemplate
from transformers import (AutoModelForSeq2SeqLM, AutoTokenizer,
                          SummarizationPipeline)


def getModelChoice():
    models = ["mistral-7b-instruct", "orca-mini-3b"]

    questions = [
        inquirer.List(
            "model",
            message="Please select a model:",
            choices=models,
        ),
    ]

    answer = inquirer.prompt(questions)

    return answer["model"]


def setupLangChain(model, batches=4, threads=8, nPredict=1024):
    path = returnModelLocalPath(model)

    # Callbacks support token-wise streaming
    callbacks = [StreamingStdOutCallbackHandler()]

    # Verbose is required to pass to the callback manager
    llm = GPT4All(
        model=path,
        callbacks=callbacks,
        verbose=True,
        n_batch=batches,
        n_threads=threads,
        n_predict=nPredict,
    )

    template = """
Here's my function in Python:

{function}

Given the definition of a function in Python, generate it's documentation. I want it complete with fields like function name, function arguments and return values as well as a detailed explanation of how the function logic works line-by-line. Make it concise and informative to put the documentation into a project.

Here's an example of how to generate the documentation for a function:

## Function Name: `log_directory_structure`

### Arguments
* `directory_path` (str): The path of the directory to be logged. It is required.
* `ai_context` (dict): A dictionary containing information about the AI context. This parameter can be used for additional functionality, but it's not necessary for logging the directory structure.

### Return Values
None - this function does not return any value.

### Explanation of Function Logic:
1. The function first checks if the given path is valid by using `os.path.exists()`. If the path is invalid, it prints an error message and returns without logging anything.
2. It then gets a list of items in the directory using `os.listdir()` and iterates through each item.
3. For each item, it checks if it's a directory by using `os.path.isdir()`. If it is, it logs the directory name using `print()`, adjusting the indentation level for subdirectories using string multiplication (`" " * indent`) and concatenating it with the current directory path.
4. It then recursively calls itself on the item path to log its subdirectory structure.
5. If the item is a file, it can be logged similarly by calling `print()` with the item name and path.
6. After logging all items in the directory, it prompts the user for the directory path using `input()` and logs the root directory using `print()`.
7. Finally, it calls itself on the given directory path to log its structure recursively.
    """

    prompt = PromptTemplate(
        template=template,
        input_variables=["function"],
    )

    llmChain = LLMChain(
        prompt=prompt,
        llm=llm,
    )

    return llmChain


def createPipeline(checkpoint, device):
    '''
    Create a transformers model summarization pipeline.

    Arguments:
    checkpoint - model checkpoint
    device (integer) - either 0 or 1, to specify if there exists a GPU
    '''
    pipeline = SummarizationPipeline(
        model=AutoModelForSeq2SeqLM.from_pretrained(checkpoint),
        tokenizer=AutoTokenizer.from_pretrained(
            checkpoint,
            skip_special_tokens=True,
            legacy=False
        ),
        max_new_tokens=1024,
        device=device
    )

    return pipeline