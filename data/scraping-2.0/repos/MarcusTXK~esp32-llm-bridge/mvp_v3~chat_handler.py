from langchain.llms import LlamaCpp
from langchain.prompts import PromptTemplate
from langchain.output_parsers import PydanticOutputParser
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from config import MODEL
from pydantic import BaseModel, Field
from langchain.agents import AgentType, Tool, initialize_agent
from langchain.memory import ConversationBufferMemory
from langchain.tools import DuckDuckGoSearchRun

class State(BaseModel):
    light: int = Field(description="1 for on, 0 for off", ge=0, le=1)
    msg: str = Field(description="Response to the user's commands")

class ChatHandler:
    def __init__(self):
        callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
        llm = LlamaCpp(
            n_ctx=1024,
            model_path=MODEL,
            temperature=0,
            # max_tokens=100,
            # top_p=1,
            f16_kv=True,  # MUST set to True, otherwise you will run into problem after a couple of calls
            callback_manager=callback_manager,
            verbose=True,  # Verbose is required to pass to the callback manager
        )
        self.parser = PydanticOutputParser(pydantic_object=State)
        memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        search = DuckDuckGoSearchRun()
        duckduckgo_tool = Tool(
            name="Search",
            func=search.run,
            description="Used to search on the internet to find information, only use if you are very unsure of how to answer.",
        )
        tools = [duckduckgo_tool]

        self.agent_executor = initialize_agent(
            tools=tools,
            llm=llm,
            agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            verbose=True,
            # memory=memory,        
            max_iterations=3,
            handle_parsing_errors=True
        )


    def send_chat(self, state, user_input):
        # template = f"""
        # The current environment data: {state}
        # """
        # prompt = PromptTemplate(
        #     template="<s>[INST]\n{format_instructions}\nNo explanations are neded other than the JSON."
        #     + template
        #     + "</s>\n[INST]{input}[/INST]",
        #     input_variables=["input"],
        #     partial_variables={"format_instructions": self.parser.get_format_instructions()},
        # )

        # _input = prompt.format_prompt(input=user_input)
        # output = self.llm(_input.to_string())

        output = self.agent_executor.invoke(
            {
                "input": user_input,
            }
        )

        print(output)
        return output['output']

        # # TODO add handler if it fails to parse
        # return self.parser.parse(output)
