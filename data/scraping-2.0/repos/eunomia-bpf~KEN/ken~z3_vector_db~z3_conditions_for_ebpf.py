import json
import os
import re
import time
from langchain.chains.conversation.memory import ConversationBufferMemory
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationChain
from langchain.chains.openai_functions import create_openai_fn_chain
from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    PromptTemplate,
)
from langchain.chains import LLMChain
from langchain.llms import DeepInfra, HuggingFaceHub

message_prompt = HumanMessagePromptTemplate(
    prompt=PromptTemplate(
        template="{input}",
        input_variables=["input"],
    )
)
prompt_template = ChatPromptTemplate.from_messages([message_prompt])


def extract_code_blocks(text: str) -> str:
	# Check if triple backticks exist in the text
	if "```" not in text:
		return text.replace("bpftrace -e", "").strip().strip("'")

	pattern = r"```(.*?)```"
	matches = re.findall(pattern, text, re.DOTALL)
	res = "\n".join(matches)
	return res.replace("bpftrace -e", "").strip().strip("'")


def run_code_llama_for_prog(question: str) -> str:
	if len(question) >= 5 * 3000:
		print("question too long, truncating to 5 * 3000 chars")
		question = question[: 5 * 3000]
	llm = DeepInfra(model_id="codellama/CodeLlama-34b-Instruct-hf")
	llm.model_kwargs = {
		"temperature": 0.7,
		"repetition_penalty": 1.2,
		"max_new_tokens": 2048,
		"top_p": 0.9,
	}

	template = """<s>[INST] <<SYS>>
	You should only write the bpftrace program itself.
	No explain and no instructions. No words other than bpftrace program.
	<</SYS>> {question} [/INST]
	"""

	prompt = PromptTemplate(template=template, input_variables=["question"])
	llm_chain = LLMChain(prompt=prompt, llm=llm)
	res = llm_chain.run(question)
	return extract_code_blocks(res)

def run_bpftrace_prog_with_func_call_define(prog: str) -> str:
    """Runs a bpftrace program. You should only input the eBPF program itself.

    Args:
                    prog: The bpftrace program to run.
    """
    return f"{prog}"

def run_gpt_for_bpftrace_func(input: str, model_name: str) -> str:
    # If we pass in a model explicitly, we need to make sure it supports the OpenAI function-calling API.
    llm = ChatOpenAI(model=model_name, temperature=0)
    chain = create_openai_fn_chain(
        [run_bpftrace_prog_with_func_call_define], llm, prompt_template, verbose=False
    )
    res = chain.run(input)
    print(res)
    prog = res["prog"]
    return prog


def libbpf_prompt(statement, doc):
    prompt = """
    I'm working on a project involving eBPF (Extended Berkeley Packet Filter) programs and I need to generate Z3 conditions for various eBPF helpers. These conditions will be used to prove the correctness of the eBPF programs. 
    The output should be in JSON format. For example:
    ```json
    {
        "bpf_map_update_elem": {
            "description": "Add or update the value of the entry associated to *key* in *map* with *value*.",
            "pre": {
                "map": "!=null",
                "key": "!=null",
                "value": "!=null",
                "flags": "in [BPF_NOEXIST, BPF_EXIST, BPF_ANY]",
                "map_type": "not in [BPF_MAP_TYPE_ARRAY, BPF_MAP_TYPE_PERCPU_ARRAY] when flags == BPF_NOEXIST"
            },
            "post": {
                "return": "in [0, negative number]"
            }
        }
    }
    ```
    """
    info = f"""
    Could you assist me generate these conditions for function `{statement}` in JSON format? The helper doc is:
    ```txt
    {doc}
    ```
    """
    return prompt + info

def bpftrace_prompt(statement, doc, is_return:True):
    prompt = """
    I'm working on a project involving eBPF (Extended Berkeley Packet Filter) programs and I need to generate Z3 conditions for various eBPF helpers. These conditions will be used to prove the correctness of the eBPF programs. 
    """+("Remember this is the kretprobe function of the bpftrace, which the post condition is the return value's constraints and the name should be kretprobe:[function name]" if is_return else  "Remember this is the kprobe function of the bpftrace, which the pre condition is the input argument's constraints and the name should be kprobe:[function name]" )+("""
    The output should be in JSON format. For example:
    ```json
    {
        "kretprobe:bpf_map_update_elem": {
            "description": "Add or update the value of the entry associated to *key* in *map* with *value*.",
            "pre": {
                "map": "!=null",
                "key": "!=null",
                "value": "!=null",
                "flags": "in [BPF_NOEXIST, BPF_EXIST, BPF_ANY]",
                "map_type": "not in [BPF_MAP_TYPE_ARRAY, BPF_MAP_TYPE_PERCPU_ARRAY] when flags == BPF_NOEXIST"
            },
        }
    }
    ```
    """ if is_return else """
    The output should be in JSON format. For example:
    ```json
    {
        "kprobe:bpf_map_update_elem": {
            "description": "Add or update the value of the entry associated to *key* in *map* with *value*.",
            "pre": {
                "map": "!=null",
                "key": "!=null",
                "value": "!=null",
                "flags": "in [BPF_NOEXIST, BPF_EXIST, BPF_ANY]",
                "map_type": "not in [BPF_MAP_TYPE_ARRAY, BPF_MAP_TYPE_PERCPU_ARRAY] when flags == BPF_NOEXIST"
            },
        }
    }
    ```
    """)
    
    info = f"""
    Could you assist me generate these conditions for function `{statement}` in JSON format? The helper doc is:
    ```txt
    {doc}
    ```
    """
    return prompt + info

def generate_response(PROMPT, model_name="gpt-4", llm_backend="openai"):
    """
    Generates a response using the passed in language model.

    Parameters:
        PROMPT (str): The input prompt for the language model.

    Returns:
        str: The generated response from the language model.
    """
    if llm_backend=="openai":
        llm = ChatOpenAI(model_name=model_name, temperature=0)
        agent_chain = ConversationChain(llm=llm, verbose=False, memory=ConversationBufferMemory())
        response = agent_chain.predict(input=PROMPT)
    elif llm_backend=="wizard":
        llm = ChatWizard(model_name=model_name, temperature=0)
        agent_chain = ConversationChain(llm=llm, verbose=False, memory=ConversationBufferMemory())
        response = agent_chain.predict(input=PROMPT)
    return response

def generate_libbpf_once():
    """
    Process libbpf help information and generate JSON format Z3 verification conditions.

    Returns:
        None
    """
    responses = []
    with open("z3_vector_db/data/bpf_helper_defs_format.json", 'r', encoding='utf-8') as file:
        data = json.load(file)
        for key, value in data.items():

            prompt = libbpf_prompt(key, value)
            response = generate_response(prompt)

            code_pattern = r'```json\n(.*?)\n```'
            json_code = re.findall(code_pattern, response, re.DOTALL)
            if not json_code:
                json_code = response

            responses += json_code
            print(json_code, '\n\n')

            with open("z3_vector_db/data/libbpf_z3.json", 'a+', encoding='utf-8') as file:
                file.write(json_code[0] + ",\n")

def generate_bpftrace_once():
    """
    Generates summaries for articles and saves them in a JSON file.

    Returns:
        None
    """
    with open("z3_vector_db/data/bpftrace_z3.json", 'w', encoding='utf-8') as file:
        file.write("[")
    responses = []
    with open("z3_vector_db/data/bpf_kprobe_def_format.json", 'r', encoding='utf-8') as file:
        data = json.load(file)
        for key, value in data.items():
            # kretprobe
            prompt = bpftrace_prompt(key, value,True)
            response = generate_response(prompt)

            code_pattern = r'```json\n(.*?)\n```'
            json_code = re.findall(code_pattern, response, re.DOTALL)
            if not json_code:
                json_code = response

            responses += json_code
            print(json_code, '\n\n')

            with open("z3_vector_db/data/bpftrace_z3.json", 'a+', encoding='utf-8') as file:
                file.write(json_code[0] + ",\n")
            # kprobe
            prompt = bpftrace_prompt(key, value,False)
            response = generate_response(prompt)
            code_pattern = r'```json\n(.*?)\n```'
            json_code = re.findall(code_pattern, response, re.DOTALL)
            if not json_code:
                json_code = response

            responses += json_code
            print(json_code, '\n\n')

            with open("z3_vector_db/data/bpftrace_z3.json", 'a+', encoding='utf-8') as file:
                file.write(json_code[0] + ",\n")
    with open("z3_vector_db/data/bpftrace_z3.json", 'a+', encoding='utf-8') as file:
        file.write("]")

def get_onefunction(content):
    """
    Generates a summary for an article.

    Args:
        content (str): The content of the article.

    Returns:
        str: The generated summary.
    """

    PROMPT = f"""
    You Are required to finish the following job, I will provide the following code:
    ```text
    {content}
    ```
    """ + \
    """
    I will need for the line generate a 
    PLEASE PROVIDE THE CONTENT IN THE FOLLOWING FORMAT:
    ```json
    [{"function1": {"pre":{"var1":"condition1","var2":"condition2"},"post":{"var1":"condition3","var4":"condition4"}}]
    ```
    Here is an example:
    ```json
    {"bpf_jit_charge_modmem":{"pre":{"X":"<100"},"post":{"Z":">1"}}}
    ```
    """

    llm = ChatOpenAI(model_name="gpt-3.5-turbo-16k", temperature=0)
    agent_chain = ConversationChain(llm=llm, verbose=False, memory=ConversationBufferMemory())
    response = agent_chain.predict(input=PROMPT)
    return response


if __name__ == "__main__":
    # generate_libbpf_once()
    generate_bpftrace_once()