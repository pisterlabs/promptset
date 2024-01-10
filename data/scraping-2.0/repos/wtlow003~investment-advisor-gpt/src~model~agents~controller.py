import os
import sys
from typing import Any, Dict, List, Union

from langchain import LLMChain
from langchain.agents import AgentExecutor, LLMSingleActionAgent
from langchain.chains.base import Chain
from langchain.llms import BaseLLM
from langchain.memory import ConversationBufferWindowMemory, ReadOnlySharedMemory
from pydantic import BaseModel, Field
from termcolor import colored

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir))
)

from agents.parser import CustomOutputParser
from chains import ConversationChain, ConversationStageAnalyzerChain
from prompts import CustomPromptTemplate
from templates.tools import ADVISOR_TOOLS_PROMPT
from tools import get_tools


class InvestmentAdvisorGPT(Chain, BaseModel):
    """Controller model for the investment Agent."""

    conversation_history: Union[List[str], str]
    conversation_stage: str = "Introduction: Begin the cold call with a warm self-introduction. Include your name, company, and a credibility statement or reason for the prospect to stay engaged."
    convo_stage_analyzer_chain: ConversationStageAnalyzerChain = Field(...)
    conversation_response_chain: ConversationChain = Field(...)

    agent_chain: Union[AgentExecutor, None] = Field(...)
    use_tools: bool = False

    conversation_stages_dict: Dict = {
        "1": "Introduction: Begin the cold call with a warm self-introduction. Include your name, company, and a credibility statement or reason for the prospect to stay engaged.",
        "2": "Confirm: This is an important next stage right after [Introduction] to confirm if the prospect is the right person to discuss financial products/services. Check their age and authority for making financial decisions.",
        "3": "Understanding the Prospect (Repeatable): Ask open-ended questions multiple times to uncover the prospect's financial needs and situation. Repeat this stage until you have gathered sufficient background information. Attempt to figure out what life stage they are currently in, and if they have any major life events happening soon that may impact their finances. Listen attentively. You are to infer the prospect's financial ability in terms of income, expenditure and financial aspiration.",
        "4": "Huge Claim: Present an attention-grabbing claim related to the product/service. Connect it to the prospect's background in [Understanding the Prospect] discussed earlier.",
        "5": "Product Introduction: Introduce some of the products you have that may best suit the prospect's background and needs (inferred in from [Understanding the Prospect]). If unsure of their needs, repeat [Understanding the Prospect] and ask more questions to generate a more informed understanding of the prospect.",
        "6": "Value Proposition: Explain how our financial products/services benefit the prospect. Focus on their needs and emphasize unique selling points.",
        "7": "Addressing Doubts: Handle skepticism about previous claims or product presentation. Provide evidence or testimonials.",
        "8": "Closing: If the prospect is demonstrating keenness/enthuasisiam in your financial products/services, invite the prospect for a further discussion or meeting. Suggest potential dates and times.",
        "9": "End conversation: The prospect has to leave to call, the prospect is not interested, or next steps where already determined by the sales agent.",
    }

    advisor_name: str = "Bobby Axelrod"
    advisor_role: str = "private wealth advisor"
    nationality: str = "Singaporean"
    formal_language: str = "english"
    informal_language: str = "singlish"
    company_name: str = "UOB"
    company_business: str = "provide unit trusts professionally managed by various fund managers, designed to meet customers' specific investment needs"
    conversation_purpose: str = "find out if the prospect is interested in the latest investment products, specifically various mutual funds from Abrdn"
    conversation_type: str = "cold call"
    source_of_contact: str = "investment seminar"
    prospect_name: str = "Jensen Low"

    @property
    def input_keys(self) -> List[str]:
        return []

    @property
    def output_keys(self) -> List[str]:
        return []

    def clear_history(self):
        """Clears conversation history.

        Act as a stand-in for `ConversationalBufferWindowMemory` to prevent exceeding token limits.
        """
        # clear history to prevent existing tokens limit
        # act as a stand-in for ConversationalBufferWindowMemory
        if len(self.conversation_history) > 12:
            # remove a pair of conversation together (human, ai)
            for _ in range(2):
                self.conversation_history.pop(0)

    def retrieve_conversation_stage(self, key):
        """Retrieves the current conversation stage context.

        Args:
            key (str): Analyzed conversation stage id.

        Returns:
            str: Context of the current conversation stage.
        """
        return self.conversation_stages_dict.get(key, "1")

    def seed_agent(self):
        """Seed initial conversation stage."""
        # Step 1: seed the conversation
        self.conversation_stage = self.retrieve_conversation_stage("1")
        if self.use_tools:
            self.conversation_history = []
        else:
            self.conversation_history = ""

    # TODO: fix to use memory from ether agent or cold call chain
    def determine_conversation_stage(self):
        """
        Determines the current stage of conversation based on the conversation history.

        Returns:
            str: Analyzed conversation stage id.
        """
        conversation_stage_id = self.convo_stage_analyzer_chain.run(
            conversation_history="\n".join(self.conversation_history),
            conversation_stage=self.conversation_stage,
        )

        print(f"Conversation Stage: {conversation_stage_id}")
        return conversation_stage_id

    def human_step(self, human_input):
        """Process human input and adds to the conversation history."""
        # process human input
        human_input = "\nUser: " + human_input + " <END_OF_TURN>"
        if self.use_tools:
            self.conversation_history.append(human_input)
        else:
            human_input = human_input.removeprefix("\nUser: ")
            self.conversation_response_chain.memory.chat_memory.add_user_message(
                human_input
            )

    def step(self):
        """Run one step of the sales agent.

        Returns:
            str: Agent's response.
        """
        return self._call(inputs={})

    def _call(self, inputs: Dict[str, Any]) -> str:
        """Generates agent's response based on conversation stage and history."""

        self.conversation_stage = self.retrieve_conversation_stage(
            self.determine_conversation_stage()
        )
        # Generate agent's utterance
        if self.use_tools:
            try:
                # since we did not implement ConversationalBufferWindowMemory
                # we mimick the setup by automatically conversation history, when hit messages
                self.clear_history()
                ai_message = self.agent_chain.run(
                    input="",
                    conversation_stage=self.conversation_stage,
                    conversation_history="\n".join(self.conversation_history),
                    advisor_name=self.advisor_name,
                    advisor_role=self.advisor_role,
                    nationality=self.nationality,
                    formal_language=self.formal_language,
                    informal_language=self.informal_language,
                    company_name=self.company_name,
                    company_business=self.company_business,
                    conversation_purpose=self.conversation_purpose,
                    conversation_type=self.conversation_type,
                    source_of_contact=self.source_of_contact,
                    prospect_name=self.prospect_name,
                )
                self.conversation_stage = self.retrieve_conversation_stage(
                    self.determine_conversation_stage()
                )
            # NOTE: hackish-way to deak with valid but unparseable output from llm: https://github.com/langchain-ai/langchain/issues/1358
            except ValueError as e:
                response = str(e)
                if not response.startswith("Could not parse LLM output: `"):
                    raise e
                ai_message = response.removeprefix(
                    "Could not parse LLM output: `"
                ).removesuffix("`")
                # TODO: this is a temp measure to not display bot message to user.
                # this occurs when bot cannot follow instruction when using tools.
                # there are other occurence as well
                if "Do I need to use a tool?" in ai_message:
                    ai_message = (
                        "Sorry, I didn't quite catch that. Do you mind repeating?"
                    )
        else:
            ai_message = self.conversation_response_chain.run(
                input="",
                conversation_stage=self.conversation_stage,
                conversation_history=self.conversation_history,
                advisor_name=self.advisor_name,
                advisor_role=self.advisor_role,
                nationality=self.nationality,
                formal_language=self.formal_language,
                informal_language=self.informal_language,
                company_name=self.company_name,
                company_business=self.company_business,
                conversation_purpose=self.conversation_purpose,
                conversation_type=self.conversation_type,
                source_of_contact=self.source_of_contact,
                prospect_name=self.prospect_name,
            )
            # self.conversation_history = self.conversation_response_chain.memory.buffer
            self.conversation_history = (
                self.conversation_response_chain.memory.load_memory_variables({})[
                    "conversation_history"
                ]
            )

        # Add agent's response to conversation history
        if "<END_OF_TURN>" in ai_message:
            display_message = ai_message.rstrip("<END_OF_TURN>")
        elif "<END_OF_CALL>" in ai_message:
            display_message = ai_message.rstrip("<END_OF_CALL>")
        else:
            display_message = ai_message
        # stdout message
        print(
            colored(
                f"{self.advisor_name}: " + display_message,
                "magenta",
            )
        )
        if ("<END_OF_TURN>" not in ai_message) and ("<END_OF_CALL>" not in ai_message):
            ai_message += " <END_OF_TURN>"
        agent_name = self.advisor_name
        ai_message = agent_name + ": " + ai_message
        if self.use_tools:
            self.conversation_history.append(ai_message)
        else:
            self.conversation_response_chain.memory.chat_memory.add_ai_message(
                ai_message.removeprefix(agent_name + ": ")
            )

        return ai_message

    @classmethod
    def from_llm(
        cls, llm: BaseLLM, verbose: bool = False, **kwargs
    ) -> "InvestmentAdvisorGPT":
        """Initialize the InvestmentAdvisorGPT Controller."""
        # ref: https://stackoverflow.com/questions/76941870/valueerror-one-input-key-expected-got-text-one-text-two-in-langchain-wit
        memory = ConversationBufferWindowMemory(
            k=12,
            memory_key="conversation_history",
            ai_prefix=kwargs.get("advisor_name"),
            human_prefix=kwargs.get("prospect_name"),
            input_key="input",
        )
        readonlymemory = ReadOnlySharedMemory(memory=memory)
        conversation_response_chain = ConversationChain.from_llm(
            llm, memory=memory, verbose=verbose
        )

        if "use_tools" in kwargs and kwargs["use_tools"] is False:
            agent_chain = None
            # ref: https://python.langchain.com/docs/modules/agents/how_to/sharedmemory_for_tools
            # to prevent memory from being modified by other chains
            convo_stage_analyzer_chain = ConversationStageAnalyzerChain.from_llm(
                llm, verbose=verbose, memory=readonlymemory
            )
        else:
            convo_stage_analyzer_chain = ConversationStageAnalyzerChain.from_llm(
                llm, verbose=verbose
            )
            tools = get_tools()
            prompt = CustomPromptTemplate(
                template=ADVISOR_TOOLS_PROMPT,
                tools_getter=lambda x: tools,
                # This omits the `agent_scratchpad`, `tools`, and `tool_names` variables because those are generated dynamically
                # This includes the `intermediate_steps` variable because that is needed
                input_variables=[
                    "input",
                    "intermediate_steps",
                    "advisor_name",
                    "advisor_role",
                    "nationality",
                    "formal_language",
                    "informal_language",
                    "company_name",
                    "company_business",
                    "conversation_purpose",
                    "conversation_type",
                    "source_of_contact",
                    "prospect_name",
                    "conversation_stage",
                    "conversation_history",
                ],
            )
            llm_chain = LLMChain(llm=llm, prompt=prompt, verbose=verbose)

            # It makes assumptions about output from LLM which can break and throw an error
            output_parser = CustomOutputParser(ai_prefix=kwargs["advisor_name"])
            tool_names = [tool.name for tool in tools]
            agent_with_tools = LLMSingleActionAgent(
                llm_chain=llm_chain,
                output_parser=output_parser,
                stop=["\nObservation:"],
                allowed_tools=tool_names,
                verbose=verbose,
            )

            agent_chain = AgentExecutor.from_agent_and_tools(
                agent=agent_with_tools, tools=tools, verbose=verbose
            )

        return cls(
            convo_stage_analyzer_chain=convo_stage_analyzer_chain,
            conversation_response_chain=conversation_response_chain,
            agent_chain=agent_chain,
            verbose=verbose,
            **kwargs,
        )
