from copy import deepcopy
from typing import Any, Dict, List

from langchain.chains.base import Chain
from langchain.llms import BaseLLM
from langchain import (
    LLMChain,
    LLMMathChain,
    SQLDatabase,
    SQLDatabaseChain
)
from langchain import PromptTemplate

from langchain.agents import ZeroShotAgent, Tool, AgentExecutor
from langchain import OpenAI, SerpAPIWrapper, LLMChain
from langchain.prompts import MessagesPlaceholder
from langchain.memory import ConversationBufferMemory
from collections import deque
from typing import Dict, List, Optional, Any

from langchain.llms import BaseLLM
from langchain.vectorstores.base import VectorStore
from pydantic import BaseModel, Field
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains.base import Chain
from langchain.agents import AgentExecutor
from pydantic import BaseModel, Field

from careai.core.chains import DentalOfficeSecretaryConversationChain, StageAnalyzerChain, TaskCreationChain, TaskPrioritizationChain
from careai.utils.logger import time_logger
from careai.core.stages import CONVERSATION_STAGES


class DentalOfficeSecretaryGPT(Chain, BaseModel):
    """Controller model for the Dental Office Secretary Agent."""

    conversation_history: List[str] = []
    conversation_stage_id: str = "1"
    current_conversation_stage: str = CONVERSATION_STAGES.get("1")
    stage_analyzer_chain: StageAnalyzerChain = Field(...)
    conversation_utterance_chain: DentalOfficeSecretaryConversationChain = Field(...)
    agent_executor: AgentExecutor = Field(...)
    conversation_stage_dict: Dict = CONVERSATION_STAGES

    dental_office_secretary_name: str = "Maggie Malloy"
    dental_office_secretary_role: str = "Dental Office Secretary"
    dental_office_name: str = "Keep them clean"
    dental_office_business: str = "Keep them clean is a premium dental office that treats patients with the utmost care and respect. We offer a wide range of services, including general dentistry, cosmetic dentistry, and orthodontics. Our goal is to provide our patients with the best possible experience from start to finish. We are committed to providing you with a comfortable environment where you can feel at ease while receiving treatment."
    dental_office_values: str = "Our mission at Keep them clean is to help people live healthier lives by providing them with quality dental care. We believe that everyone deserves access to affordable dental care, regardless of their income or insurance status. That's why we offer a variety of payment options for our patients, including cash, credit cards, and CareCredit financing. We also accept most major insurance plans, so you don't have to worry about paying out-of-pocket for your visit. If you're looking for a dentist who will treat you like family, look no further than Keep them clean!"
    conversation_purpose: str = "Serve the patient by answering their questions and helping them with their needs such as scheduling an appointment or getting information about our services."
    conversation_type: str = "call"

    def retrieve_conversation_stage(self, key):
        return self.conversation_stage_dict.get(key, "1")

    @property
    def input_keys(self) -> List[str]:
        return []

    @property
    def output_keys(self) -> List[str]:
        return []

    @time_logger
    def seed_agent(self):
        # Step 1: seed the conversation
        self.current_conversation_stage = self.retrieve_conversation_stage("1")
        self.conversation_history = []

    @time_logger
    def determine_conversation_stage(self):
        self.conversation_stage_id = self.stage_analyzer_chain.run(
            conversation_history="\n".join(self.conversation_history).rstrip("\n"),
            conversation_stage_id=self.conversation_stage_id,
            conversation_stages="\n".join(
                [
                    str(key) + ": " + str(value)
                    for key, value in CONVERSATION_STAGES.items()
                ]
            ),
        )

        print(f"Conversation Stage ID: {self.conversation_stage_id}")
        self.current_conversation_stage = self.retrieve_conversation_stage(
            self.conversation_stage_id
        )

        print(f"Conversation Stage: {self.current_conversation_stage}")

    def human_step(self, human_input):
        # process human input
        human_input = "User: " + human_input + " <END_OF_TURN>"
        self.conversation_history.append(human_input)

    @time_logger
    def step(self, return_streaming_generator: bool = False):
        """
        Args:
            return_streaming_generator (bool): whether or not return
            streaming generator object to manipulate streaming chunks in downstream applications.
        """
        if not return_streaming_generator:
            self._call(inputs={})
            # Check if the agent need a tool
            if '<CALL_FOR_TOOLS>' in self.conversation_history[-1]:
                human_input = self.conversation_history[-2]
                self._call_for_tools({"objective": human_input})
        else:
            return self._streaming_generator()

    # TO-DO change this override "run" override the "run method" in the SalesConversation chain!
    @time_logger
    def _streaming_generator(self):
        """
        Sometimes, the agent wants to take an action before the full LLM output is available.
        For instance, if we want to do text to speech on the partial LLM output.

        This function returns a streaming generator which can manipulate partial output from an LLM
        in-flight of the generation.

        Example:

        >> streaming_generator = self._streaming_generator()
        # Now I can loop through the output in chunks:
        >> for chunk in streaming_generator:
        Out: Chunk 1, Chunk 2, ... etc.
        See: https://github.com/openai/openai-cookbook/blob/main/examples/How_to_stream_completions.ipynb
        """
        prompt = self.conversation_utterance_chain.prep_prompts(
            [
                dict(
                    conversation_stage=self.current_conversation_stage,
                    conversation_history="\n".join(self.conversation_history),
                    dental_office_secretary_name=self.dental_office_secretary_name,
                    dental_office_secretary_role=self.dental_office_secretary_role,
                    dental_office_name=self.dental_office_name,
                    dental_office_business=self.dental_office_business,
                    dental_office_values=self.dental_office_values,
                    conversation_purpose=self.conversation_purpose,
                    conversation_type=self.conversation_type,
                )
            ]
        )

        inception_messages = prompt[0][0].to_messages()

        message_dict = {"role": "system", "content": inception_messages[0].content}

        if self.conversation_utterance_chain.verbose:
            print("\033[92m" + inception_messages[0].content + "\033[0m")
        messages = [message_dict]

        return self.conversation_utterance_chain.llm.completion_with_retry(
            messages=messages,
            stop="<END_OF_TURN>",
            stream=True,
            model="gpt-3.5-turbo-0613",
        )

    def _call(self, inputs: Dict[str, Any]) -> None:
        """Run one step of the agent."""

        # Generate agent's utterance
        ai_message = self.sales_conversation_utterance_chain.run(
            conversation_stage=self.current_conversation_stage,
            conversation_history="\n".join(self.conversation_history),
            dental_office_secretary_name=self.dental_office_secretary_name,
            dental_office_secretary_role=self.dental_office_secretary_role,
            dental_office_name=self.dental_office_name,
            dental_office_business=self.dental_office_business,
            dental_office_values=self.dental_office_values,
            conversation_purpose=self.conversation_purpose,
            conversation_type=self.conversation_type,
        )

        # Add agent's response to conversation history
        agent_name = self.dental_office_secretary_name
        ai_message = agent_name + ": " + ai_message
        self.conversation_history.append(ai_message)
        print(ai_message.replace("<END_OF_TURN>", ""))
        return {}

    def _call_for_tools(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Run the agent tool helper."""
        context = inputs["context"]
        objective = inputs["objective"]
        first_task = inputs.get("first_task", "Make a todo list")
        self.add_task({"task_id": 1, "task_name": first_task})
        num_iters = 0
        while True:
            if self.task_list:
                self.print_task_list()

                # Step 1: Pull the first task
                task = self.task_list.popleft()
                self.print_next_task(task)

                # Step 2: Execute the task
                result = execute_task(
                    self.vectorstore, self.execution_chain, objective, task["task_name"]
                )
                this_task_id = int(task["task_id"])
                self.print_task_result(result)

                # Step 3: Store the result in Pinecone
                result_id = f"result_{task['task_id']}"
                self.vectorstore.add_texts(
                    texts=[result],
                    metadatas=[{"task": task["task_name"]}],
                    ids=[result_id],
                )

                # Step 4: Create new tasks and reprioritize task list
                new_tasks = get_next_task(
                    self.task_creation_chain,
                    result,
                    task["task_name"],
                    [t["task_name"] for t in self.task_list],
                    objective,
                )
                for new_task in new_tasks:
                    self.task_id_counter += 1
                    new_task.update({"task_id": self.task_id_counter})
                    self.add_task(new_task)
                self.task_list = deque(
                    prioritize_tasks(
                        self.task_prioritization_chain,
                        this_task_id,
                        list(self.task_list),
                        objective,
                    )
                )
            num_iters += 1
            if self.max_iterations is not None and num_iters == self.max_iterations:
                print(
                    "\033[91m\033[1m" + "\n*****TASK ENDING*****\n" + "\033[0m\033[0m"
                )
                break
        return {}

    @classmethod
    @time_logger
    def from_llm(cls, llm: BaseLLM, verbose: bool = False, **kwargs) -> "DentalOfficeSecretaryGPT":
        """Initialize the DentalOfficeSecretaryGPT Controller."""
        stage_analyzer_chain = StageAnalyzerChain.from_llm(llm, verbose=verbose)

        if (
            "use_custom_prompt" in kwargs.keys()
            and kwargs["use_custom_prompt"] == "True"
        ):
            use_custom_prompt = deepcopy(kwargs["use_custom_prompt"])
            custom_prompt = deepcopy(kwargs["custom_prompt"])

            # clean up
            del kwargs["use_custom_prompt"]
            del kwargs["custom_prompt"]

            conversation_utterance_chain = DentalOfficeSecretaryConversationChain.from_llm(
                llm,
                verbose=verbose,
                use_custom_prompt=use_custom_prompt,
                custom_prompt=custom_prompt,
            )

        else:
            conversation_utterance_chain = DentalOfficeSecretaryConversationChain.from_llm(
                llm, verbose=verbose
            )

        # Add reflexion chain and tools
        llm_math_chain = LLMMathChain.from_llm(llm=llm, verbose=True)
        db = SQLDatabase.from_uri("postgresql://valentin:margera@localhost:5433/postgres", include_tables=['patients', 'dentists', 'appointments'], sample_rows_in_table_info=2)

        query_response_limit = 1
        db_chain = SQLDatabaseChain.from_llm(llm, db, verbose=True, top_k=query_response_limit, use_query_checker=True)

        tools = [
            Tool(
                name="Calculator",
                func=llm_math_chain.run,
                description="useful for when you need to answer questions about math",
            ),
            Tool(
                name="DentistDB",
                func=db_chain.run,
                description="useful for when you need add an appointment, modify it or remove it or add a patient, modify it or remove it or seeing the relation between the patient the doctor and the appoitments. Input should be in the form of a question containing full context",
            ),
        ]
        llm_chain = LLMChain(llm=llm, prompt=prompt)
        tool_names = [tool.name for tool in tools]
        agent = ZeroShotAgent(llm_chain=llm_chain, allowed_tools=tool_names)
        agent_executor = AgentExecutor.from_agent_and_tools(
            agent=agent, tools=tools, verbose=True
        )

        return cls(
            stage_analyzer_chain=stage_analyzer_chain,
            conversation_utterance_chain=conversation_utterance_chain,
            agent_executor=agent_executor,
            verbose=verbose,
            **kwargs,
        )
