import typer
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chains import LLMChain
from langchain.llms import GPT4All
from langchain.prompts import PromptTemplate
from transformers import (AutoModelForSeq2SeqLM, AutoTokenizer,
                          SummarizationPipeline)
from yaspin import yaspin

from utils import *

app = typer.Typer()
spinner = yaspin()


@app.command()
def generate(
    file: str,
    default: bool = True,
    gpu: bool = True
):
    '''
    Typer command to generate the documentation for an input script file.

    Arguments:
    file (string) - path to the input file
    default (boolean) - flag to specify using the default model or a custom model
    gpu (boolean) - flag to specify whether a GPU is available
    '''
    print(f"Arguments specified:")
    print(f"File Path: {file}")
    print(f"Use Default Model Flag: {default}")
    print(f"Use GPU: {gpu}")

    # Change gpu flag
    device = 0 if gpu else 1

    # Implement logic to switch models given a different language
    checkpoint = r"SEBIS/code_trans_t5_base_source_code_summarization_python_transfer_learning_finetune" if default else -1

    print("\nCreating model summarization pipeline...\n")
    spinner.start()
    pipeline = model_utils.createPipeline(checkpoint=checkpoint, device=device)
    spinner.stop()

    code = file_utils.readFile(file)
    tokenizedCode = pre_process_utils.pythonTokenizer(code)

    print(f"Code:\n\n{code}")
    print(f"\n\nCode after tokenization:\n\n{tokenizedCode}")
    print(f"\n\nModel Output through inference point:\n\n{pipeline([tokenizedCode])}")


@app.command()
def easter():
    '''
    An easter egg for the keen eyed open-source contributor.
    '''
    print("This is a test command! If you're here, that means you've discovered something pretty cool in our code-base.\nCongrats and thanks for using our app!")


if __name__ == "__main__":
    app()