# Have two models converse with each other
# Specialized to Mistral (prompts are different)

# Without a grammar, controlling the output is not possible.
# Each large language model (LLM) generates a complete conversation.
# However, by specifying that each LLM represents a single author responding to another, the output appears to be more controlled.

import os
import re
import generic_utils as u
from langchain.callbacks.manager import CallbackManager
from pprint import pprint

# from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

# from langchain.chains import LLMChain
from langchain.llms import LlamaCpp

# from langchain.prompts import PromptTemplate
# from llama_cpp.llama import Llama
from llama_cpp.llama_grammar import LlamaGrammar

# not used on mac
os.environ["CUBLAS_LOGINFO_DBG"] = "1"

# Callbacks support token-wise streaming
# callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
callback_manager = CallbackManager([])
# ----------------------------------------------------------------------

n_gpu_layers = 20  # Change this value based on your model and your GPU VRAM pool.
n_batch = 256  # Should be between 1 and n_ctx, consider the amount of VRAM in your GPU.

LLM_MODELS = os.environ["LLM_MODELS"]

data, rendered_texts = u.apply_templates("scenario_hawking_smolin.yaml")
pprint(rendered_texts)

with open(data["grammar_file"], "r") as file:
    grammar_text = file.read()
grammar = LlamaGrammar.from_string(grammar_text)


# Make sure the model path is correct for your system!
def myLlamaCpp(model: str):
    """
    Create an instance of LlamaCpp with the given model path.

    Args:
        model (str): The path to the LlamaCpp model.

    Returns:
        LlamaCpp: An instance of the LlamaCpp class.
    """
    # No idea how to set these parameters relating to repetition and frequency
    llm = LlamaCpp(
        model_path=model,
        n_gpu_layers=n_gpu_layers,
        # repeat_penalty=5.2,
        # repeat_last_n=5,
        # frequency_penalty=0.5,
        # presence_penalty=0.5,
        n_ctx=4096,
        stop=[],
        # stop=["</s>"],  # If used, the message will stop early
        max_tokens=500,
        n_threads=8,
        temperature=2.0,  # also works with 0.0 (0.01 is safer)
        f16_kv=True,
        n_batch=n_batch,
        callback_manager=callback_manager,
        verbose=False,
        grammar_path="json_only_reply.gbnf",
    )
    return llm


modelA = LLM_MODELS + "mistral-7b-instruct-v0.1.Q3_K_M.gguf"
modelB = LLM_MODELS + "mistral-7b-instruct-v0.1.Q3_K_M.gguf"

llmA = myLlamaCpp(modelA)
llmB = myLlamaCpp(modelB)

authorA = data["authors"]["authorA"]
authorB = data["authors"]["authorB"]
subject = data["conversation"]["subject"]
output_format = data["conversation"]["output_format"]
additional_context_authorA = data["conversation"]["additional_context_authorA"]
additional_context_authorB = data["conversation"]["additional_context_authorB"]
instructions = data["instructions"]

msgsA = u.MistralMessages(subject, additional_context_authorA)
msgsB = u.MistralMessages(subject, additional_context_authorB)

msgsA.add(authorA, "")


conversation = u.Conversation(
    instructions,
    subject,
    authorA,
    authorB,
    llmA,
    llmB,
    msgsA,
    msgsB,
    output_format,
    grammar,
)
conversation.multi_turn(10)
