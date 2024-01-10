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
from langchain.agents import load_tools
from langchain.agents import initialize_agent
from typing import List
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM
from transformers import AutoTokenizer, AutoModelForCausalLM
from accelerate import infer_auto_device_map, init_empty_weights
import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb=128'

# This was inserted by me
import torch
# Check if CUDA is available and set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)
print(torch.cuda.get_device_name(0))
print(torch.cuda.is_available())
torch.cuda.empty_cache()
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
    # small -> upstage/llama-30b-instruct-2048
    # with access -> garage-bAInd/Platypus2-70B-instruct
    #   'meta-llama/Llama-2-70b-chat-hf'
    # vicuna -> lmsys/vicuna-13b-delta-v1.1
    # bigscience/bloom-560m
    model_id ='bigscience/bloom-560m'
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
    
    model_config = transformers.AutoConfig.from_pretrained(model_id, use_auth_token=hf_token, trust_remote_code=True)
    # config = AutoConfig.from_pretrained("upstage/llama-30b-instruct-2048", trust_remote_code=True)
    # with init_empty_weights():
    #     model = AutoModelForCausalLM.from_config(config, trust_remote_code = True)

    # device_map = infer_auto_device_map(model, no_split_module_classes=["OPTDecoderLayer"], dtype="float16")
    
    # device_map["model.decoder.layers.37"] = "disk"

    
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_id,
        trust_remote_code=True,
        # torch_dtype=torch.float16,
        config=model_config,
        # offload_folder="offload",
        # offload_state_dict = True,
        # quantization_config=bnb_config,
        # llm_int8_enable_fp32_cpu_offload=True,
        use_auth_token=hf_token,
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
        do_sample=True,
        temperature=0.1,  # 'Randomness' of outputs, 0.0 is the min and 1.0 the max
        max_new_tokens=512,  # Max number of tokens to generate in the output
        repetition_penalty=1.1  # Without this, output begins repeating
    )

    llm = HuggingFacePipeline(pipeline=generate_text)
    memory = ConversationBufferWindowMemory(
    memory_key="chat_history", k=5, return_messages=True, output_key="output"
    )
    tools = load_tools(["llm-math"], llm=llm)

    
    # To output the agent's responses --
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
    # --

    
    
    # print(agent.run("Who is the United States President? What is his current age raised divided by 2?"))

    # initialize agent
    # agent = initialize_agent(
    #     agent="chat-conversational-react-description",
    #     tools=tools,
    #     llm=llm,
    #     verbose=False,
    #     early_stopping_method="generate",
    #     memory=memory,
    #     #agent_kwargs={"output_parser": parser}
    # )
    # agent.agent.llm_chain.prompt

    # We need to add special tokens used to signify the beginning and end of instructions, and the beginning and end of system messages.
    # # These are described in the Llama-2 model cards on Hugging Face.
    # B_INST, E_INST = "[INST]", "[/INST]"
    # B_SYS, E_SYS = "<>\n", "\n<>\n\n"


    # sys_msg = B_SYS + """Act as an support engineer in a software company""" + E_SYS

    # new_prompt = agent.agent.create_prompt(
    #     system_message=sys_msg,
    #     tools=tools
    # )
    # agent.agent.llm_chain.prompt = new_prompt

    # instruction = B_INST + " Respond to the following in list of steps and ask the user if they want steps to automate it further" + E_INST
    # human_msg = instruction + "\nUser: {input}"

    # agent.agent.llm_chain.prompt.messages[2].prompt.template = human_msg
    # answer2 = agent.agent.llm_chain.prompt
    # Generate text based on the input query
    context = """Managed Spot Training can be used with all instances
    supported in Amazon SageMaker. Managed Spot Training is supported
    in all AWS Regions where Amazon SageMaker is currently available."""

    prompt_template = """Answer the following QUESTION based on the CONTEXT
    given. Answer the questions as if you are providing steps to resolve the issue. If you do not know the answer and the CONTEXT doesn't
    contain the answer truthfully say "Looks like the issue is complicated. Please escalate this issue to L2 engineers.".

    CONTEXT:
    {context}

    QUESTION:
    {question}

    ANSWER:
    """
    text_input = prompt_template.replace("{context}", context).replace("{question}", query)

    

    # Fetch Context
    llm = HuggingFacePipeline(pipeline=generate_text)

    embeddings = HuggingFaceEmbeddings(model_name=embeddings_model_name)
    print("DB work started")
    db = Chroma(persist_directory=persist_directory, embedding_function=embeddings, client_settings=CHROMA_SETTINGS)
    print("db work ended")
    retriever = db.as_retriever()
    docs = db.similarity_search(query)
    print("-------this is docs",docs)
    contexts = [doc.page_content for doc in docs]
    print("-------contexts", contexts)

    max_section_len = 1000
    separator = "\n"

    def construct_context(contexts: List[str]) -> str:
        chosen_sections = []
        chosen_sections_len = 0

        for text in contexts:
            text = text.strip()
            # Add contexts until we run out of space.
            chosen_sections_len += len(text) + 2
            if chosen_sections_len > max_section_len:
                break
            chosen_sections.append(text)
        concatenated_doc = separator.join(chosen_sections)
        print(
            f"With maximum sequence length {max_section_len}, selected top {len(chosen_sections)} document sections: \n{concatenated_doc}"
        )
        return concatenated_doc
    context_str = construct_context(contexts=contexts)
    
    print(context_str)

    text_input = prompt_template.replace("{context}", context_str).replace("{question}", query)

    print("-------text input  \n", text_input)
    # agent = initialize_agent(tools, 
    #                      llm, 
    #                      agent="zero-shot-react-description", 
    #                      verbose=False,
    #                      agent_kwargs={"output_parser": parser})

    rag_pipeline = RetrievalQA.from_chain_type(
    llm=llm, chain_type='stuff',
    retriever=retriever, return_source_documents=False
    )

    try:
        # Generate text based on the input query
        answer = rag_pipeline(query)
        # answer = agent.run(text_input)
        print("\n ------------    this is gpt\n",answer)
        return answer
    except Exception as e:
        # Handle any exceptions that may occur during the execution of the agent
        # print(f"Error while running the agent: {e}")
        return {'error': str(e)}
