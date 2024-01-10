"""Chain that carries on a conversation and calls an LLM."""
from typing import Dict, List, Any, Optional
from pydantic import Extra, Field

from langchain.callbacks.manager import CallbackManagerForChainRun
from langchain.chat_models import ChatOpenAI
from langchain.chat_models.base import BaseChatModel
from langchain.memory import ChatMessageHistory
from langchain.chains.llm import LLMChain
from langchain.prompts.base import BasePromptTemplate
from langchain.schema import BaseChatMessageHistory
from langchain.schema import SystemMessage


class ChatChain(LLMChain):
    """
    Chain to have a conversation and maintain conversation history for Chat type models

    Example:
        .. code-block:: python

            chat_chain = ChatChain(prompt=PromptTemplate(input_variables=["foo"], template="Say {foo}")
    """
    prompt: BasePromptTemplate
    system_message: str = None
    history: BaseChatMessageHistory = Field(default_factory=ChatMessageHistory)
    llm: BaseChatModel = Field(default_factory=ChatOpenAI)

    output_key: str = "response"  #: :meta private:

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # initialize system message
        if self.system_message:
            self.history.add_message(SystemMessage(content=self.system_message))

    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.forbid
        arbitrary_types_allowed = True

    @property
    def input_keys(self) -> List[str]:
        """Use this since so some prompt vars come from history."""
        return self.prompt.input_variables

    @property
    def output_keys(self) -> List[str]:
        """Will always return text key.

        :meta private:
        """
        return [self.output_key]

    def _call(
            self,
            inputs: Dict[str, Any],
            run_manager: Optional[CallbackManagerForChainRun] = None,
    ) -> Dict[str, str]:
        # format human message
        prompt_value = self.prompt.format(**inputs)

        # add to history
        self.history.add_user_message(prompt_value)

        # Call chat llm
        response = self.llm(
            self.history.messages,
            callbacks=run_manager.get_child() if run_manager else None
        )

        # add response to history
        self.history.add_ai_message(response.content)

        # log results to run manager
        if run_manager:
            run_manager.on_text(f"history: {self.history.messages[:-2]}\nprompt: {prompt_value}\nresponse: {response.content}")

        return {self.output_key: response.content}


    @property
    def _chain_type(self) -> str:
        return "ChatChain"


if __name__ == '__main__':
    # Test usage
    from dotenv import load_dotenv
    from langchain import PromptTemplate
    from langchain.callbacks import StdOutCallbackHandler
    from langchain.memory import ChatMessageHistory
    from langchain.schema import SystemMessage

    load_dotenv()

    chat = ChatOpenAI(
        model_name="gpt-3.5-turbo",
        temperature=0,
        verbose=True
    )

    system_message = """I would like you to recite the colors of the rainbow to the user. The list of colors, in order, should be: "red", "orange", "yellow", "green", "blue", "indigo", "violet".

    However, please note that the user may not always ask for all seven colors at once. Instead, they may ask for a specific number of colors. When this happens, your task is to recite the requested number of colors from the list, starting from "red" and proceeding in order.

    If the user asks for more colors at a later point, you should continue from where you last stopped. For example, if the user first requests three colors, you should say "red", "orange", "yellow". If the user then asks for two more colors, you should continue with "green", "blue".

    You should remember the last color you recited and continue the sequence from there each time the user requests more colors. However, once you recite "violet", if the user requests more colors, you should start back from "red".
    The user has requested 0 colors so the next one will be red."""

    prompt = PromptTemplate.from_template("""Give me the next {number} colors""")

    llmchain = ChatChain(llm=chat, prompt=prompt, callbacks=[StdOutCallbackHandler()],
                         system_message=system_message, verbose=True)

    response = llmchain.run(number=3)

    print(response)


    # memory.buffer.append(prompt.format_messages(number=2))
    response = llmchain.run(number=2)

    print(response)
