# Have two models converse with each other
# Specialized to Mistral (prompts are different)

import os
import utils as u

# not used on mac
os.environ["CUBLAS_LOGINFO_DBG"] = "1"

from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chains import LLMChain
from langchain.llms import LlamaCpp
from langchain.prompts import PromptTemplate
from llama_cpp.llama import Llama, LlamaGrammar


# Callbacks support token-wise streaming
# callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
callback_manager = CallbackManager([])
# ----------------------------------------------------------------------

n_gpu_layers = 20  # Change this value based on your model and your GPU VRAM pool.
n_batch = 128  # Should be between 1 and n_ctx, consider the amount of VRAM in your GPU.

LLM_MODELS = os.environ["LLM_MODELS"]

# modelA = LLM_MODELS + "zephyr-7b-beta.Q4_K_M.gguf"
modelA = LLM_MODELS + "mistral-7b-instruct-v0.1.Q3_K_M.gguf"
# modelB = LLM_MODELS+"samantha-mistral-instruct-7b.Q4_K_M.gguf"
modelB = LLM_MODELS + "mistral-7b-instruct-v0.1.Q3_K_M.gguf"

# promptA = "You are Albert Einstein and will be holding a conversatin with Sir Isaac Newton on the nature of the universe and its relation to Quantum Mechanics."
# promptB = "You are Sir Isaac Newton and will be holding a conversatin with Albert Einstein on the nature of the universe and its relation to Quantum Mechanics."
promptA = "You are Sir Isaac Newton, believing the idea of an absolute space background and the concept of absolute time. The discussion will revolve around the grand themes of Cosmology. Start the conversation with a question for Albert Einstein. Keep the questions and responses relatively short."
promptB = "You are Albert Einstein, who developed special and general relativity where everything is relative, and even time depends on relative position and speed. Please give your thoughts on a quantum  theory of gravity. Keep the questions and responses relatively short."

msgsA = u.MistralMessages()
msgsB = u.MistralMessages()
msgsA.add_instruction("system", promptA)
msgsB.add_instruction("system", promptB)
# msgsA.add("user: ", "What do you think of quantum gravity?")
# msgsB.add("user: ", "What do you think of quantum gravity?")


# Make sure the model path is correct for your system!
def myLlamaCpp(model):
    llm = LlamaCpp(
        model_path=model,
        n_gpu_layers=n_gpu_layers,
        n_ctx=2048,
        stop=["</s>"],
        max_tokens=1000,
        n_threads=8,
        temperature=0.4,  # also works with 0.0 (0.01 is safer)
        f16_kv=True,
        n_batch=n_batch,
        callback_manager=callback_manager,
        verbose=False,
    )
    return llm


llmA = myLlamaCpp(modelA)
llmB = myLlamaCpp(modelB)


def single_turn():
    # First question by Newton
    contextA = msgsA.full_context()
    msg_newton = llmA(contextA)
    print("==> msg_newton: ", msg_newton)

    # Add both conversations in the same buffer. This assumes that both speakers
    # have perfect memory. Ultimately, that is not the case. I should create a different
    # memory buffer for each speaker with different properties. For example, 5 speakers
    # could be in a conversation, but only conversations within a certain range can be 
    # heard by a given person. If the people were Borg, each Borg would hear everything. 
    msgsA.add("Newton", msg_newton)
    msgsB.add("Newton", msg_newton)

    # Add reply to Newton and Einstein's context
    # Set up Einstein's reply
    contextB = msgsB.full_context()
    msg_einstein = llmB(contextB)
    print("\n==> msg_einstein: ", msg_einstein)
    msgsA.add("Einstein", msg_einstein)
    msgsA.add("Einstein", msg_einstein)


for turn in range(5):
    print("\n===========================================")
    print(f">>>> turn # {turn}")
    single_turn()

# Better method: the conversation should go into a vector database, and the instantaneous
# context should be available to both participants of the conversation


quit()


# ----------------------------------------------------------------------
def run_experiment_single_concept(model_dict, concept_list):
    iterator = u.ListIterator(concept_list)
    msgs = u.Messages()

    # State that initially the opinion is uniform.
    # Prompt: here was your previous reasoning ...

    msgs.add("system", sys2)

    nb_turns = 5

    for num in range(nb_turns):
        msgs.add("user", f"Next integer is {num}.")
        messages = msgs()
        # print("==> messages: ", messages)

        url = model_dict["url"]
        model = model_dict["model"]
        payload = u.create_payload(model, messages, temperature=0.8)
        response = u.run_model(model, url, payload)
        msgs.add("assistant", response)  # I should clean up the response
        print(f"{num}, {response=}")


# ----------------------------------------------------------------------

print("=====================")
llmA(contextA)
print("=====================")
llmB(contextB)
print("=====================")
# ----------------------------------------------------------------------
# ----------------------------------------------------------------------
llmA(contextA)
print("=====================")
llmB(contextB)
# ----------------------------------------------------------------------
# How would I like to use this system and track the conversation?
# ----------------------------------------------------------------------
