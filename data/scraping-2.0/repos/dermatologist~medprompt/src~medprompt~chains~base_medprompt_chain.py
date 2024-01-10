from kink import di
import re
from langchain.tools import StructuredTool
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
from .. import MedPrompter

class BaseMedpromptChain:

    class ChainInput(BaseModel):
        question: str = Field()

    def __init__(self,
                 chain=None,
                 prompt={},
                 main_llm=None,
                 clinical_llm=None,
                 sec_llm=None,
                 input_type=None,
                 output_type=None
                ):
        self._chain = chain
        self._prompt = prompt
        self._main_llm = main_llm
        self._clinical_llm = clinical_llm
        self._sec_llm = sec_llm
        self._input_type = input_type
        self._output_type = output_type
        self._name = self.get_name()
        self.med_prompter = MedPrompter()
        self._description = self.get_name()
        self.init_prompt()

    @property
    def chain(self):
        if self._chain is None:
            """Get the runnable chain."""
            """ RunnableParallel / RunnablePassthrough / RunnableSequential / RunnableLambda / RunnableMap / RunnableBranch """
            _cot = RunnablePassthrough.assign(
                question = lambda x: x["question"],
                ) | self.prompt | self.main_llm | StrOutputParser()
            chain = _cot.with_types(input_type=self.input_type)
            return chain

    @property
    def prompt(self):
        return self._prompt

    @property
    def main_llm(self):
        if self._main_llm is None:
            self._main_llm = di["main_llm"]
        return self._main_llm

    @property
    def clinical_llm(self):
        if self._clinical_llm is None:
            self._clinical_llm = di["clinical_llm"]
        return self._clinical_llm

    @property
    def sec_llm(self):
        if self._sec_llm is None:
            self._sec_llm = di["sec_llm"]
        return self._sec_llm

    @property
    def input_type(self):
        if self._input_type is None:
            self._input_type = self.ChainInput
        return self._input_type

    @property
    def output_type(self):
        return self._output_type

    @property
    def name(self):
        return self._name

    @property
    def description(self):
        return self._description

    @chain.setter
    def chain(self, value):
        self._chain = value

    @prompt.setter
    def prompt(self, value):
        self._prompt = value
        self.init_prompt()

    @main_llm.setter
    def main_llm(self, value):
        self._main_llm = value

    @clinical_llm.setter
    def clinical_llm(self, value):
        self._clinical_llm = value

    @sec_llm.setter
    def sec_llm(self, value):
        self._sec_llm = value

    @input_type.setter
    def input_type(self, value):
        self._input_type = value

    @output_type.setter
    def output_type(self, value):
        self._output_type = value

    @name.setter
    def name(self, value):
        self._name = value

    @description.setter
    def description(self, value):
        self._description = value

    def invoke(self, **kwargs):
        return self.chain.invoke(kwargs)

    def __call__(self, **kwargs):
        return self.invoke(**kwargs)

    def get_name(self):
        return re.sub(r'(?<!^)(?=[A-Z])', '_', self.__class__.__name__).lower()

    @DeprecationWarning
    def get_runnable(self, **kwargs):
        return self.chain

    #* Override these methods in subclasses
    def init_prompt(self):
        pass

    # @classmethod
    # def get_tool(cls, **kwargs):
    #     """Get the tool."""
    #     return StructuredTool.from_function(
    #         func=cls(**kwargs).chain.invoke,
    #         name=cls(**kwargs).name,
    #         description=cls(**kwargs).description,
    #         args_schema=cls(**kwargs).input_type,
    #         # coroutine= ... <- you can specify an async method if desired as well
    #     ).run(kwargs)
