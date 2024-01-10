import os, uuid, shutil, re
from PIL import Image
from loguru import logger
from models.chatglm import ChatGLM
from utils import AutoConfiguration
from langchain.tools import Tool
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.agents.initialize import initialize_agent
from langchain.agents import AgentType, load_tools
from prompts.en_prompt import PREFIX, DOC_GPT_FORMAT_INSTRUCTIONS, DOC_GPT_SUFFIX


@AutoConfiguration("configs/init_config.yaml")
class ToolMatrix():

    def __init__(self) -> None:
        self.config = None
        self.tools = []
        # self.llm = ChatOpenAI(temperature=0.1,
        #                       model_name="gpt-3.5-turbo",
        #                       max_tokens=1024)
        self.llm = None
        self.memory = ConversationBufferMemory(memory_key="chat_history",
                                               output_key="output")
        self.agent = None

    def init_all(self):
        self.llm = self.init_llm()
        self.init_tools()
        self.init_logger()
        self.agent = self.init_agent()

    def init_tools(self):
        for tool_info in self.config.tools:
            tool_module = tool_info["module"]
            tool_class = tool_info["class"]
            tool_cls = getattr(__import__("tools", fromlist=[tool_module]),
                               tool_class)
            arguments = [
                self.config.__dict__[tool_module][k]
                for k in self.config.__dict__[tool_module].keys()
            ] if hasattr(self.config, tool_module) else []
            tool = tool_cls(*arguments, llm=self.llm)

            if hasattr(tool, "get_tools"):
                self.tools.extend(tool.get_tools())
            else:
                self.tools.append(
                    Tool(name=tool.inference.name,
                         description=tool.inference.desc,
                         func=tool.inference))
            logger.debug(f"Tool [{tool_module}] initialized.")

        preset_tools = load_tools(self.config.preset_tools, llm=self.llm)
        logger.debug(f"[{self.config.preset_tools}] preset tools loaded.")
        self.tools.extend(preset_tools)

        logger.info(f"{len(self.tools)} tools initialized.")

    def init_llm(self):
        model_type = self.config.model_type
        if model_type == "chatglm":
            # self.llm = ChatGLM(self.config.model_config[model_type]['base_url'],
            #                    self.config.model_config[model_type]['api_key'])
            llm = ChatGLM()
            llm.load_config(self.config.model_config[model_type])
        elif model_type == "openai":
            llm = ChatOpenAI(self.config.model_config[model_type]['temperature'],
                                  self.config.model_config[model_type]['model_name'],
                                  self.config.model_config[model_type]['max_tokens'])
        else:
            raise NotImplementedError(f"Model type [{model_type}] not supported.")
        
        logger.debug(f"LLM [{model_type}] initialized.")
        return llm


    def init_agent(self):
        self.memory.clear()
        agent = initialize_agent(
            tools=self.tools,
            llm=self.llm,
            memory=self.memory,
            agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
            verbose=True,
            return_intermediate_steps=True,
            max_iterations=8,
            # 暂时先用英文的prompt
            agent_kwargs={
                "prefix": PREFIX,
                "suffix": DOC_GPT_SUFFIX,
                "format_instructions": DOC_GPT_FORMAT_INSTRUCTIONS,
            })
        logger.debug("Agent initialized.")
        return agent

    def init_logger(self):
        logger.level("INFO")

    def run_text(self, text: str, state):
        result = self.agent({"input": text.strip()})
        result['output'] = result['output'].replace("\\", "/")
        response = re.sub('(image/[-\w]*.png)',
                          lambda m: f'![](file={m.group(0)})*{m.group(0)}*',
                          result['output'])
        state = state + [(text, response)]
        logger.info(f"User input: {text}")
        logger.info(f"AI output: {response}")
        return state, state

    def run_img(self, img_path: str, state: list):
        img_path = img_path.name
        # move img to specified path
        logger.debug(f"User input a image which is saved at {img_path}.")
        # move to self.configs.image_cache_dir
        target_path = os.path.join(self.config.image_cache_dir,
                                   f"{str(uuid.uuid4())[:8]}.png")
        img = Image.open(img_path)
        img.save(target_path, format="png")
        img_name = f"image/{os.path.basename(target_path)}"
        logger.debug(f"Image moved to {img_name}.")
        # run
        HUMAN_PROMPT = f"provide a figure named {img_name}. you should use tools to finish following tasks, rather than directly imagine from my description. If you understand, say \"Received\"."
        AI_PROMPT = "Received."
        self.agent.memory.save_context({"input": HUMAN_PROMPT},
                                       {"output": AI_PROMPT})
        # self.agent.memory.buffer = self.agent.memory.buffer + HUMAN_PROMPT + "AI: " + AI_PROMPT

        state += [(f"![](file={img_name})*{img_name}*", AI_PROMPT)]

        return state, state
