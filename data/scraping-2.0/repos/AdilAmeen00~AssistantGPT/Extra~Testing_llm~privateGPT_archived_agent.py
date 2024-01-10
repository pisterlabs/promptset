from dotenv import load_dotenv
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.vectorstores import Chroma
from langchain.llms import GPT4All, LlamaCpp
import os
import xformers
from langchain.llms import HuggingFacePipeline
from langchain.memory import ConversationBufferWindowMemory
from langchain.agents import load_tools
from langchain.agents import AgentOutputParser
from langchain.agents.conversational_chat.prompt import FORMAT_INSTRUCTIONS
from langchain.output_parsers.json import parse_json_markdown
from langchain.schema import AgentAction, AgentFinish
from langchain.agents import initialize_agent
from typing import Union

# This was inserted by me
import torch
# Check if CUDA is available and set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)
print(torch.cuda.get_device_name(0))
print(torch.cuda.is_available())

load_dotenv()

embeddings_model_name = os.environ.get("EMBEDDINGS_MODEL_NAME")
persist_directory = os.environ.get('PERSIST_DIRECTORY')

model_type = os.environ.get('MODEL_TYPE')
model_path = os.environ.get('MODEL_PATH')
model_n_ctx = os.environ.get('MODEL_N_CTX')

from constants import CHROMA_SETTINGS

from torch import cuda, bfloat16
import transformers

def run_model(query):
    #   'meta-llama/Llama-2-70b-chat-hf'
    model_id = 'bigscience/bloom-560m'
    device = f'cuda:{cuda.current_device()}' if cuda.is_available() else 'cpu'

    # Set quantization configuration to load large model with less GPU memory  - Cannot use quantization in Windows
    # bnb_config = transformers.BitsAndBytesConfig(
    #     load_in_4bit=True,
    #     bnb_4bit_quant_type='nf4',
    #     bnb_4bit_use_double_quant=True,
    #     bnb_4bit_compute_dtype=bfloat16
    # )

    # Initialize model configuration and model
    hf_token = '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'
    model_config = transformers.AutoConfig.from_pretrained(model_id, use_auth_token=hf_token)
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_id,
        trust_remote_code=True,
        config=model_config,
        # quantization_config=bnb_config,
        device_map='auto'
    )
    model.eval()

    # Initialize tokenizer
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_id)

    # Define text generation pipeline
    generate_text = transformers.pipeline(
        model=model, tokenizer=tokenizer,
        return_full_text=True,  # Langchain expects the full text
        task='text-generation',
        temperature=1.0,  # 'Randomness' of outputs, 0.0 is the min and 1.0 the max
        max_new_tokens=512,  # Max number of tokens to generate in the output
        repetition_penalty=1.1  # Without this, output begins repeating
    )

    llm = HuggingFacePipeline(pipeline=generate_text)
    memory = ConversationBufferWindowMemory(
    memory_key="chat_history", k=5, return_messages=True, output_key="output"
    )
    tools = load_tools(["llm-math"], llm=llm)

    class OutputParser(AgentOutputParser):
        def get_format_instructions(self) -> str:
            return FORMAT_INSTRUCTIONS

        def parse(self, text: str) -> Union[AgentAction, AgentFinish]:
            try:
                # this will work IF the text is a valid JSON with action and action_input
                response = parse_json_markdown(text)
                action, action_input = response["action"], response["action_input"]
                if action == "Final Answer":
                    # this means the agent is finished so we call AgentFinish
                    return AgentFinish({"output": action_input}, text)
                else:
                    # otherwise the agent wants to use an action, so we call AgentAction
                    return AgentAction(action, action_input, text)
            except Exception:
                # sometimes the agent will return a string that is not a valid JSON
                # often this happens when the agent is finished
                # so we just return the text as the output
                return AgentFinish({"output": text}, text)

        @property
        def _type(self) -> str:
            return "conversational_chat"
    # initialize output parser for agent
    parser = OutputParser()

    # initialize agent
    agent = initialize_agent(
        agent="chat-conversational-react-description",
        tools=tools,
        llm=llm,
        verbose=False,
        early_stopping_method="generate",
        memory=memory,
        #agent_kwargs={"output_parser": parser}
    )
    # agent.agent.llm_chain.prompt

    # We need to add special tokens used to signify the beginning and end of instructions, and the beginning and end of system messages.
    # These are described in the Llama-2 model cards on Hugging Face.
    B_INST, E_INST = "[INST]", "[/INST]"
    B_SYS, E_SYS = "<>\n", "\n<>\n\n"


    sys_msg = B_SYS + """Act as an support engineer in a software company""" + E_SYS

    new_prompt = agent.agent.create_prompt(
        system_message=sys_msg,
        tools=tools
    )
    agent.agent.llm_chain.prompt = new_prompt

    instruction = B_INST + " Respond to the following in list of steps and ask the user if they want steps to automate it further" + E_INST
    human_msg = instruction + "\nUser: {input}"

    agent.agent.llm_chain.prompt.messages[2].prompt.template = human_msg
    answer2 = agent.agent.llm_chain.prompt
    # Generate text based on the input query
    try:
        # Generate text based on the input query
        answer = agent(query)
        print("this is gpt",answer)
        return answer
    except Exception as e:
        # Handle any exceptions that may occur during the execution of the agent
        # print(f"Error while running the agent: {e}")
        return {e}
