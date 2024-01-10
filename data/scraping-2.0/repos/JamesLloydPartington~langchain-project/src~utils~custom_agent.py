import uuid
import logging
import sys
from inspect import signature
from pydantic import BaseModel, Field

from langchain.output_parsers import PydanticOutputParser
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate


# logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
logger = logging.getLogger()


class CustomTool:

    def __init__(self,
        function: callable,
        description: str,
    ):
        
        # add a uuid to the function name
        function_name: str = function.__qualname__ + "_" + str(uuid.uuid4())[0:8]

        
        input_args = list(signature(function).parameters.keys())
        if len(input_args) != 1:
            raise Exception(f"Function must take exactly one argument, but takes {len(input_args)} arguments")
        
        input_model = signature(function).parameters[input_args[0]].annotation

        output_model = function.__annotations__["return"]
        
        # check if input_model is a pydantic model
        if not issubclass(input_model, BaseModel):
            raise Exception(f"Input model must be a pydantic model, but is {input_model}")
        
        # check if output_model is a pydantic model
        if not issubclass(output_model, BaseModel):
            raise Exception(f"Output model must be a pydantic model, but is {output_model}")


        self.function = function
        self.function_name = function_name
        self.description = description
        self.input_model: BaseModel = input_model
        self.output_model: BaseModel = output_model

    def __call__(self, *args, **kwargs):
        return self.function(*args, **kwargs)
    
    def describe(self, 
        show_function_name: bool = True,
        show_description: bool = True,
        show_input_model: bool = True,
        show_output_model: bool = True,
        ) -> str:

        description = "-"*10 + "\n"
        if show_function_name:
            description += f"function_name: '{self.function_name}'\n"
        if show_description:
            description += f"description: {self.description}\n"
        if show_input_model:
            description += f"input pydantic json_schema: {self.input_model.schema_json()}\n"
        if show_output_model:
            description += f"output pydantic json_schema: {self.output_model.schema_json()}\n"
        description += "-"*10 + "\n"
        return description

class Instruction:
    def __init__(self, text: str, tool: CustomTool):
        self.text = text
        self.tool = tool

    def run(self, llm: OpenAI, query: str, context: str):

        # turn the context into the tool's input model

        template = '''
        Given the tool/function to be used:
        {tool}

        Given the original query:
        {query}

        Given the context:
        {context}

        We are now want to run the following instruction:
        {instruction}

        Given the following information, context, query and instruction we want to predict, put all the information needed to run the tool/function into the following input model:
        {input_model}

        Response:
        '''

        parser = PydanticOutputParser(pydantic_object = self.tool.input_model)

        prompt_template = PromptTemplate(
            template=template,
            input_variables=["context" , "query", "instruction", "tool"],
            partial_variables = {"input_model": parser.get_format_instructions()},
        )

        logger.info(f"Running instruction: {self.text}")

        template_str = prompt_template.format(context=context, query=query, instruction=self.text, tool=self.tool.describe())

        logger.info(f"Template String: {template_str}")

        input_model_str = llm.predict(template_str)

        logger.info(f"Input Model: {input_model_str}")

        # turn the input model into the output model
        try:
            input_model = parser.parse(input_model_str)
        except Exception as e:
            logger.debug(f"Error parsing input model: {input_model_str}")
            logger.error(f"Error parsing input model: {e}")
            raise
        
        result = self.tool(input_model)

        logger.info(f"Result: {result}")

        return str(result)

class SystemOutput:
    def __init__(self, text: str):
        self.text = text

class HumanMessage:
    def __init__(self, text: str):
        self.text = text

class HumanSystemMessagePair:
    def __init__(self, human_message: HumanMessage, system_message: SystemOutput):
        self.human_message = human_message
        self.system_message = system_message

class ChatHistory:
    def __init__(self, chat_history: list[HumanSystemMessagePair] | None = None, incomplete_human_message: HumanMessage | None = None):

        if chat_history is None:
            chat_history = []
        self.chat_history: list[HumanSystemMessagePair] = chat_history

        self._incomplete_human_message: HumanMessage | None = incomplete_human_message

    def add_human_message(self, human_message: HumanMessage):
        self._incomplete_human_message = human_message
    
    def add_system_message(self, system_message: SystemOutput):
        if self._incomplete_human_message is None:
            raise Exception("Cannot add a system message without a human message")
        self.chat_history.append(HumanSystemMessagePair(self._incomplete_human_message, system_message))
        self._incomplete_human_message = None

    def show_history(self) -> str:
        history = ""
        for pair in self.chat_history:
            history += f"Human: {pair.human_message.text}\n"
            history += f"System: {pair.system_message.text}\n"
        return history

class DeterministicPlannerAndExecutor:
    '''
    Given a query, this planner will return a list of instructions to execute. Cannot handle dynamic workflows.
    '''
    def __init__(self, tools: list[CustomTool], llm: OpenAI, history: ChatHistory | None = None):
        self.tools = tools
        self.llm = llm

        if history is None:
            history = ChatHistory()
        self.history = history

        self.tools_by_name = {tool.function_name: tool for tool in self.tools}

    def show_tools(self) -> str:
        description = ""
        for tool in self.tools:
            description += tool.describe()
        return description


    def create_plan(self, query: str) -> list[Instruction]:

        class InstructionOutput(BaseModel):
            action: str = Field(description="A single action to take with a tool. Suggest information, context and instructions are needed to run the action.")
            function_name: str = Field(description="The name of the function to call (the function_name).")

        class PlanOutput(BaseModel):
            instructions: list[InstructionOutput] = Field(description="A list of instructions to execute which will complete the task. An instruction cannot do more than one thing.")
        
        template = '''
        Given the following tools:
        {tools}

        Given the chat history:
        {history}

        The following human query is:
        {query}


        Create a plan to execute the query. A plan is a list of instructions,
        an instruction is a single action to take using exactly one tool (exactly one function_name).
        In the action be precise about what you are looking for. 
        Make the instructions will acomplish all the tasks in the query.
        Provide a list of actions using tools in the format of:
        {instruction}

        Response:
        '''

        parser = PydanticOutputParser(pydantic_object = PlanOutput)

        prompt_template = PromptTemplate(
            template=template,
            input_variables=["tools", "query", "history"],
            partial_variables={"instruction": parser.get_format_instructions()},
        )

        instructions_str = self.llm.predict(prompt_template.format(tools=self.show_tools(), query=query, history=self.history.show_history()))

        logger.info(instructions_str)

        try:
            instructions = parser.parse(instructions_str)
        except Exception as e:
            logger.error(f"Error parsing instructions: {e}")
            raise
        
        instruction_list: list[Instruction] = []
        for instruction in instructions.instructions:
            tool = self.tools_by_name[instruction.function_name]
            instruction_list.append(Instruction(text=instruction.action, tool=tool))

        return instruction_list
    
    def predict(self, query: str):
        instructions = self.create_plan(query=query)

        for instruction in instructions:
            logger.info('-'*100)
            logger.info(f"Running instruction: {instruction.text}")
        
        instruction_outputs = self.history.show_history()

        for i, instruction in enumerate(instructions):

            instruction_output = instruction.run(llm=self.llm, query=query, context=instruction_outputs)

            instruction_outputs += f"The answer to the question '{instruction.text}' is:\n {instruction_output}\n"

        prompt = f'''
        Given the original query:
        {query}

        Given the history:
        {self.history.show_history()}

        Given the following outputs for subsequent instructions:
        {instruction_outputs}

        Give a final answer to all the questions asked in the query:
        '''

        llm_output = self.llm.predict(prompt)

        self.history.add_human_message(HumanMessage(query))
        self.history.add_system_message(SystemOutput(llm_output))
        
        return llm_output

    


