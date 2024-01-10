from typing import List
from typing import Optional

from langchain.callbacks.manager import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)
from langchain.chains.llm import LLMChain
from langchain.chat_models.base import BaseChatModel
from langchain.experimental.autonomous_agents.autogpt.output_parser import (
    AutoGPTOutputParser,
    BaseAutoGPTOutputParser,
)
from langchain.experimental.autonomous_agents.autogpt.prompt import AutoGPTPrompt
from langchain.experimental.autonomous_agents.autogpt.prompt_generator import (
    FINISH_NAME,
)
from langchain.schema import (
    AIMessage,
    BaseMessage,
    Document,
    HumanMessage,
    SystemMessage,
)
from langchain.tools.base import BaseTool
from langchain.vectorstores.base import VectorStoreRetriever
from mergedbots import MergedMessage, MergedBot
from mergedbots.experimental.sequential import ConversationSequence
from pydantic import ValidationError


class HumanInputRun(BaseTool):
    """Tool that adds the capability to ask user for input."""

    name = "Human"
    description = (
        "You can ask a human for guidance when you think you "
        "got stuck or you are not sure what to do next. "
        "The input should be a question for the human."
    )

    bot: MergedBot
    conv_sequence: ConversationSequence
    latest_inbound_msg: MergedMessage

    def _run(
        self,
        query: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Use the Human input tool."""
        raise NotImplementedError("`MergedBots` human tool does not support synchronous mode")

    async def _arun(
        self,
        query: str,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> str:
        """Use the Human tool asynchronously."""
        await self.send_feedback(query, is_still_typing=False)
        self.latest_inbound_msg = await self.conv_sequence.wait_for_incoming()
        return self.latest_inbound_msg.content

    async def send_feedback(self, feedback: str, is_still_typing=True) -> None:
        """Send a message to the user."""
        await self.conv_sequence.yield_outgoing(
            await self.latest_inbound_msg.bot_response(
                self.bot, feedback, is_still_typing=is_still_typing, is_visible_to_bots=True
            )
        )


class AutoGPT:
    """Agent class for interacting with Auto-GPT."""

    def __init__(
        self,
        ai_name: str,
        memory: VectorStoreRetriever,
        chain: LLMChain,
        output_parser: BaseAutoGPTOutputParser,
        tools: List[BaseTool],
        feedback_tool: HumanInputRun,
    ):
        self.ai_name = ai_name
        self.memory = memory
        self.full_message_history: List[BaseMessage] = []
        self.next_action_count = 0
        self.chain = chain
        self.output_parser = output_parser
        self.tools = tools
        self.feedback_tool = feedback_tool

    @classmethod
    def from_llm_and_tools(
        cls,
        ai_name: str,
        ai_role: str,
        memory: VectorStoreRetriever,
        tools: List[BaseTool],
        llm: BaseChatModel,
        feedback_tool: Optional[HumanInputRun],
        output_parser: Optional[BaseAutoGPTOutputParser] = None,
    ) -> "AutoGPT":
        prompt = AutoGPTPrompt(
            ai_name=ai_name,
            ai_role=ai_role,
            tools=tools,
            input_variables=["memory", "messages", "goals", "user_input"],
            token_counter=llm.get_num_tokens,
        )
        chain = LLMChain(llm=llm, prompt=prompt)
        return cls(
            ai_name,
            memory,
            chain,
            output_parser or AutoGPTOutputParser(),
            tools,
            feedback_tool,
        )

    async def arun(self, goals: List[str]) -> str:
        user_input = "Determine which next command to use, " "and respond using the format specified above:"
        # Interaction Loop
        loop_count = 0
        while True:
            # Discontinue if continuous limit is reached
            loop_count += 1

            # Send message to AI, get response
            assistant_reply = await self.chain.arun(
                goals=goals,
                messages=self.full_message_history,
                memory=self.memory,
                user_input=user_input,
            )

            # Print Assistant thoughts
            await self.feedback_tool.send_feedback(f"```json\n{assistant_reply}\n```")

            # TODO replace this history with the mergedbots history ?
            self.full_message_history.append(HumanMessage(content=user_input))
            self.full_message_history.append(AIMessage(content=assistant_reply))

            # Get command name and arguments
            action = self.output_parser.parse(assistant_reply)
            tools = {t.name: t for t in self.tools}
            if action.name == FINISH_NAME:
                if self.feedback_tool is not None:
                    await self.feedback_tool.send_feedback(
                        f"FINISHED: {action.args['response']}", is_still_typing=False
                    )
                return action.args["response"]

            tool = None
            if action.name in tools:
                tool = tools[action.name]
                try:
                    observation = await tool.arun(action.args)
                except ValidationError as e:
                    observation = f"Validation Error in args: {str(e)}, args: {action.args}"
                except Exception as e:
                    observation = f"Error: {str(e)}, {type(e).__name__}, args: {action.args}"
                result = f"Command {tool.name} returned: {observation}"
            elif action.name == "ERROR":
                result = f"Error: {action.args}. "
            else:
                result = (
                    f"Unknown command '{action.name}'. "
                    f"Please refer to the 'COMMANDS' list for available "
                    f"commands and only respond in the specified JSON format."
                )

            memory_to_add = f"Assistant Reply: {assistant_reply} " f"\nResult: {result} "
            if not isinstance(tool, HumanInputRun):
                await self.feedback_tool.send_feedback(result)
                feedback = await self.feedback_tool.arun("Input: ")
                if feedback in {"q", "stop"}:
                    await self.feedback_tool.send_feedback("EXITING", is_still_typing=False)
                    return "EXITING"
                memory_to_add += f"\n{feedback}"

            self.memory.add_documents([Document(page_content=memory_to_add)])
            self.full_message_history.append(SystemMessage(content=result))
