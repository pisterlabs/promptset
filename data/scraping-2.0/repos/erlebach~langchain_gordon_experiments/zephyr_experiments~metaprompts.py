from langchain.chains import LLMChain
from langchain.llms import OpenAI
from langchain.memory import ConversationBufferWindowMemory
from langchain.prompts import PromptTemplate

import os
from langchain.llms import LlamaCpp
from langchain.callbacks.manager import CallbackManager


LLM_MODELS = os.environ["LLM_MODELS"]

model_path = LLM_MODELS + "mistral-7b-instruct-v0.1.Q3_K_M.gguf"
n_gpu_layers = 12
n_batch = 128
callback_manager = CallbackManager([])


def myLlamaCpp(model: str):
    """
    Create an instance of LlamaCpp with the given model path.

    Args:
        model (str): The path to the LlamaCpp model.

    Returns:
        LlamaCpp: An instance of the LlamaCpp class.
    """
    llm = LlamaCpp(model_path=model,
        n_gpu_layers=n_gpu_layers,
        n_ctx=4096,
        stop=[],
        # stop=["</s>"],  # If used, the message will stop early
        max_tokens=1000,
        n_threads=8,
        temperature=2.0,  # also works with 0.0 (0.01 is safer)
        f16_kv=True,
        n_batch=n_batch,
        callback_manager=callback_manager,
        verbose=False,
    )
    return llm


def initialize_chain(instructions, memory=None):
    if memory is None:
        memory = ConversationBufferWindowMemory()
        print("memory: ", memory)
        memory.ai_prefix = "Assistant"

    template = f"""
    Instructions: {instructions}
    {{{memory.memory_key}}}
    Human: {{human_input}}
    Assistant:"""

    prompt = PromptTemplate(
        input_variables=["history", "human_input"], template=template
    )

    
    llm1 = myLlamaCpp(model_path)

    chain = LLMChain(
        # llm=OpenAI(temperature=0),
        llm = llm1,
        prompt=prompt,
        verbose=False,
        memory=ConversationBufferWindowMemory(),
    )
    return chain


def initialize_meta_chain():
    meta_template = """
    Assistant has just had the below interactions with a User. Assistant followed their "Instructions" closely. Your job is to critique the Assistant's performance and then revise the Instructions so that Assistant would quickly and correctly respond in the future.

    ####

    {chat_history}

    ####

    Please reflect on these interactions.

    You should first critique Assistant's performance. What could Assistant have done better? What should the Assistant remember about this user? Are there things this user always wants? Indicate this with "Critique: ...".

    You should next revise the Instructions so that Assistant would quickly and correctly respond in the future. Assistant's goal is to satisfy the user in as few interactions as possible. Assistant will only see the new Instructions, not the interaction history, so anything important must be summarized in the Instructions. Don't forget any important details in the current Instructions! Indicate the new Instructions by "Instructions: ...".
    """

    meta_prompt = PromptTemplate(
        input_variables=["chat_history"], template=meta_template
    )

    llm2 = myLlamaCpp(model_path)

    meta_chain = LLMChain(
        llm=llm2,
        prompt=meta_prompt,
        verbose=True,
    )
    return meta_chain


def get_chat_history(chain_memory):
    memory_key = chain_memory.memory_key
    chat_history = chain_memory.load_memory_variables(memory_key)[memory_key]
    return chat_history


def get_new_instructions(meta_output):
    delimiter = "Instructions: "
    new_instructions = meta_output[meta_output.find(delimiter) + len(delimiter) :]
    return new_instructions



def main(task, max_iters=3, max_meta_iters=5):
    failed_phrase = "task failed"
    success_phrase = "task succeeded"
    key_phrases = [success_phrase, failed_phrase]

    instructions = "None"
    for i in range(max_meta_iters):
        print(f"[Episode {i+1}/{max_meta_iters}]")
        chain = initialize_chain(instructions, memory=None)
        output = chain.predict(human_input=task)
        for j in range(max_iters):
            print(f"(Step {j+1}/{max_iters})")
            print(f"Assistant: {output}")
            print("Enter human input: ")
            human_input = input()
            if any(phrase in human_input.lower() for phrase in key_phrases):
                break
            output = chain.predict(human_input=human_input)
        if success_phrase in human_input.lower():
            print("You succeeded! Thanks for playing!")
            return
        meta_chain = initialize_meta_chain()
        meta_output = meta_chain.predict(chat_history=get_chat_history(chain.memory))
        print(f"Feedback: {meta_output}")
        instructions = get_new_instructions(meta_output)
        print(f"New Instructions: {instructions}")
        print("\n" + "#" * 80 + "\n")
    print("You failed! Thanks for playing!")

#----------------------------------------------------------------------
if __name__ == "__main__":
    task = "Provide a systematic argument for why we should always eat pasta with olives."
    main(task)
