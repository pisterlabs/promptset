import transformers
from transformers import AutoTokenizer
from langchain.llms import HuggingFacePipeline
from langchain.agents import load_tools, initialize_agent, AgentType
from langchain import PromptTemplate
from langchain.chains import LLMChain, SimpleSequentialChain
from langchain.output_parsers import StructuredOutputParser
from modules.roles_templates.roles_templates import roles_template
from modules.schemas.brain_schema import response_schemas
import torch

model = "meta-llama/Llama-2-7b-chat-hf"
tokenizer = AutoTokenizer.from_pretrained(model)

pipe = transformers.pipeline(
    "text-generation",
    model=model, 
    tokenizer=tokenizer, 
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    device_map="auto",
    max_length=1000,
    do_sample=True,
    temperature=0.5,
    top_p=0.95,
    repetition_penalty=1.2,
    eos_token_id=tokenizer.eos_token_id
)

local_llm = HuggingFacePipeline(pipeline=pipe)
tools = load_tools(["google-serper"], llm=local_llm )
output_parser = StructuredOutputParser.from_response_schemas(response_schemas)

class AssistantLlama2:
    def __create_chain(self):
        prompt_template = roles_template.get("prompt_template")
        prompt = PromptTemplate(
            template=prompt_template,
            input_variables=["input"]
        )
        return LLMChain(prompt=prompt, llm=local_llm)
    
    def __create_agent(self):
        return initialize_agent(
            tools=tools,
            llm=local_llm,
            agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            verbose=True
        )
    
    def chat(self, input):
        agent = self.__create_agent()
        chain = self.__create_chain()
        overall_chain = SimpleSequentialChain(
            chains=[agent, chain],
            verbose=True
        )
        response = overall_chain.run(input)
        return output_parser.parse(response)