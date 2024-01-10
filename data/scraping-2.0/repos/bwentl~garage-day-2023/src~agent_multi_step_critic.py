import sys
import os
import re
import warnings
import random
from typing import Any, List, Optional, Type, Dict

from langchain.agents import (
    initialize_agent,
    Tool,
    AgentType,
)

sys.path.append("./")
# from src.gradio_ui import WebUI
from src.models import LlamaModelHandler
from src.agent_tool_selection import AgentToolSelection
from src.docs import DocumentHandler
from src.tools import ToolHandler
from src.memory_store import PGMemoryStoreSetter, PGMemoryStoreRetriever
from src.util import get_secrets, get_word_match_list, agent_logs
from src.prompts.tool_select import TOOL_SELECTION_PROMPT
from src.prompts.multi_step import (
    MULTI_STEP_TOOL_FOLLOW_UP_PROMPT,
    MULTI_STEP_TOOL_PICKER_PROMPT,
    MULTI_STEP_TOOL_USER_PROMPT,
    MULTI_STEP_TOOL_CRITIC_EVIDENCE_PROMPT,
    MULTI_STEP_TOOL_GENERATE_PROMPT,
    MULTI_STEP_TOOL_CRITIC_PROMPT,
)


# suppress warnings for demo
warnings.filterwarnings("ignore")
os.environ["PYDEVD_INTERRUPT_THREAD_TIMEOUT"] = "60"
os.environ["PYDEVD_WARN_EVALUATION_TIMEOUT "] = "60"


class AgentMultiStepCritic:
    """
    A qa answering agent that is self critical and adaptive in collecting additional information
    similar to agent executor by langchain, the goal is to obtain answer based on factual information
    collected using various tools.
    However, this qa answering agent will perform information gathering through tools in multiple steps and will
    constantly assess the needs to obtain additional information and judging the value of its preliminary actions
    The benefit of this agent type is that it will provide more well formed and thoughful response.
    """

    def __init__(
        self,
        pipeline,
        embedding,
        tool_names,
        doc_info=dict(),
        run_tool_selector=False,
        update_long_term_memory=False,
        use_long_term_memory=False,
        verbose=False,
        **kwarg,
    ):
        self.kwarg = kwarg
        self.pipeline = pipeline
        self.embedding = embedding
        self.new_session = kwarg["new_session"] if "new_session" in kwarg else False
        self.generate_search_term = (
            kwarg["generate_search_term"] if "generate_search_term" in kwarg else True
        )
        self.min_tool_use = kwarg["min_tool_use"] if "min_tool_use" in kwarg else 1
        self.max_tool_use = kwarg["max_tool_use"] if "max_tool_use" in kwarg else 3
        self.use_cache_from_log = (
            kwarg["use_cache_from_log"] if "use_cache_from_log" in kwarg else False
        )
        self.update_long_term_memory = update_long_term_memory
        self.use_long_term_memory = use_long_term_memory
        self.long_term_memory_collection = (
            kwarg["long_term_memory_collection"]
            if "long_term_memory_collection" in kwarg
            else "long_term"
        )
        self.run_tool_selector = run_tool_selector
        self.log_tool_selector = (
            kwarg["log_tool_selector"] if "log_tool_selector" in kwarg else True
        )
        doc_use_type = kwarg["doc_use_type"] if "doc_use_type" in kwarg else "stuff"
        doc_top_k_results = (
            kwarg["doc_top_k_results"] if "doc_top_k_results" in kwarg else 3
        )
        # build tools
        tools_wrapper = ToolHandler()
        tools = tools_wrapper.get_tools(tool_names, pipeline)
        # add document retrievers to tools
        if len(doc_info) > 0:
            newDocs = DocumentHandler(
                embedding=embedding, redis_host=get_secrets("redis_host")
            )
            doc_tools = newDocs.get_tool_from_doc(
                pipeline=pipeline,
                doc_info=doc_info,
                doc_use_type=doc_use_type,
                doc_top_k_results=doc_top_k_results,
            )
            tools = tools + doc_tools
        # initialize memory bank
        if self.update_long_term_memory or self.use_long_term_memory:
            self.memory_setter = self._init_long_term_memory_setter(embedding)
            memory_tool = self._init_long_term_memory_retriver(embedding)
            if self.use_long_term_memory:
                tools.append(memory_tool)
        # make tools avilable to instance
        self.tools = tools

    def run(self, main_prompt):
        tools = self.tools
        if self.new_session:
            # set cache state to save cache logs
            cached_response = agent_logs.set_cache_lookup(
                f"Agent Multi Step Critic - {main_prompt}"
            )
            # if using cache from logs saved, then try to load previous log
            if cached_response is not None and self.use_cache_from_log:
                return cached_response
            agent_logs().clear_log()

        # initiate agent executor chain
        if self.run_tool_selector:
            tools = AgentToolSelection(
                pipeline=self.pipeline,
                tools=self.tools,
                **self.kwarg,
            )

        # print shortlist of tools being used
        tool_list = ToolHandler.get_tools_list(tools)
        tool_list_display = f"Tools available: {tool_list}"
        agent_logs.write_log_and_print(tool_list_display)
        # print question
        display_header = "\x1b[1;32m" + f"""\n\nQuestion: {main_prompt}""" + "\x1b[0m"
        agent_logs.write_log_and_print(display_header)

        # entering the information gathering loop
        enough_info = False
        tools_used = list()
        previous_tool_output = ""
        preliminary_answer = ""
        previous_follow_up_questions = ""
        number_of_tries = 0
        while enough_info == False and number_of_tries < self.max_tool_use:
            # step 1: ask follow up question
            follow_up_question_prompt = (
                MULTI_STEP_TOOL_FOLLOW_UP_PROMPT.replace("{main_prompt}", main_prompt)
                .replace(
                    "{previous_tool_output}",
                    "nothing yet"
                    if previous_tool_output == ""
                    else previous_tool_output,
                )
                .replace(
                    "{previous_follow_up_questions}",
                    "nothing yet"
                    if previous_follow_up_questions == ""
                    else previous_follow_up_questions,
                )
            )
            print(follow_up_question_prompt)
            follow_up_question = self.pipeline(follow_up_question_prompt)
            follow_up_question = re.sub(
                r'[^\x00-\x7f"]', "", follow_up_question
            ).strip()
            previous_follow_up_questions = (
                f"{previous_follow_up_questions}...{follow_up_question}"
            )
            agent_logs.write_log_and_print(
                f"Thought: {follow_up_question}"
                if preliminary_answer == ""
                else f"Thought: This is what I know now: {preliminary_answer}.\nI have a follow up question: {follow_up_question}"
            )
            # step 2: pick a tool or action
            tool_list_prompt = ToolHandler.get_tools_list_descriptions(tools)
            tool_picker_prompt = (
                MULTI_STEP_TOOL_PICKER_PROMPT.replace("{main_prompt}", main_prompt)
                .replace("{follow_up_question}", follow_up_question)
                .replace("{tool_list_prompt}", tool_list_prompt)
                .replace(
                    "{tools_used}",
                    "nothing yet" if len(tools_used) == 0 else str(tools_used),
                )
            )
            print(tool_picker_prompt)
            tool_picked = self.pipeline(tool_picker_prompt)
            tool_picked = re.sub(r'[^\x00-\x7f"]', "", tool_picked).strip()
            agent_logs.write_log_and_print(f"Action: {tool_picked}")
            # handling error with picking tool
            tool_bool_processor = [tool_picked.lower() in i.lower() for i in tool_list]
            tool_picked_index = (
                tool_bool_processor.index(True)
                if True in tool_bool_processor
                else random.randint(1, len(tool_list))
            )
            current_tool = tools[tool_picked_index]
            if current_tool.name not in tools_used:
                tools_used.append(current_tool.name)
            # step 3: pick an input for the tool
            if self.generate_search_term:
                tool_user_prompt = (
                    MULTI_STEP_TOOL_USER_PROMPT.replace(
                        "{tool_name}", current_tool.name
                    )
                    .replace("{tool_description}", current_tool.description)
                    .replace(
                        "{follow_up_question}",
                        follow_up_question
                        if preliminary_answer == ""
                        else f"{follow_up_question}\n\nYou already know this: {preliminary_answer}.\n",
                    )
                    .replace("{main_prompt}", main_prompt)
                )
                print(tool_user_prompt)
                tool_input = self.pipeline(tool_user_prompt)
                # remove quotation mark and non unicode characters
                tool_input = re.sub(r'[^\x00-\x7f"]', "", tool_input).strip()
                # stripe any text after period
                tool_input = tool_input.split(".")[0]
            else:
                tool_input = follow_up_question
            agent_logs.write_log_and_print(f"Action Input: {tool_input}")
            # step 4: run the tool
            tool_output = current_tool(tool_input)
            agent_logs.write_log_and_print(f"Observation: {tool_output}")
            # step 5: generate an answer
            generation_prompt = (
                MULTI_STEP_TOOL_GENERATE_PROMPT.replace("{tool_output}", tool_output)
                .replace(
                    "{previous_tool_output}",
                    "nothing yet"
                    if previous_tool_output == ""
                    else previous_tool_output,
                )
                .replace("{main_prompt}", main_prompt)
            )
            print(generation_prompt)
            preliminary_answer = self.pipeline(generation_prompt)
            # step 6: critique the tool output
            evidence_critic_prompt = MULTI_STEP_TOOL_CRITIC_EVIDENCE_PROMPT.replace(
                "{main_prompt}", main_prompt
            ).replace("{tool_output}", tool_output)
            print(evidence_critic_prompt)
            critic_evidence_output = self.pipeline(evidence_critic_prompt)
            print(critic_evidence_output)
            # if the tool output is bad, do not include it in the previous tool output
            if critic_evidence_output.lower() == "yes":
                previous_tool_output = (
                    tool_output
                    if previous_tool_output == ""
                    else f"{previous_tool_output}...{tool_output}"
                )
            # step 7: critique the generated response
            if number_of_tries < self.min_tool_use:
                critique_answer = "no"
            else:
                critique_prompt = (
                    MULTI_STEP_TOOL_CRITIC_PROMPT.replace(
                        "{preliminary_answer}", preliminary_answer
                    )
                    .replace("{main_prompt}", main_prompt)
                    .replace("{previous_tool_output}", previous_tool_output)
                )
                print(critique_prompt)
                critique_answer = self.pipeline(critique_prompt)
                print(critique_answer)
            # increment tries
            number_of_tries += 1
            # step 8: decide if final answer is ready
            if critique_answer.lower() == "yes" or number_of_tries >= self.max_tool_use:
                enough_info = True
                final_answer = preliminary_answer
                memory_thought = agent_logs.read_log()
                agent_logs.write_log_and_print(f"Final Answer: {preliminary_answer}")

        # if self.update_long_term_memory:
        #     self.memory_setter.add_memory(
        #         text=final_answer,
        #         prompt=main_prompt,
        #         thought=memory_thought,
        #         llm=self.pipeline,
        #     )

        # always cache the current log
        agent_logs.save_cache()

        return final_answer

    def _init_long_term_memory_retriver(self, embedding):
        memory_bank = PGMemoryStoreRetriever(
            embedding, self.long_term_memory_collection
        )
        memory_tool = Tool(
            name="Memory",
            func=memory_bank.retrieve_memory_list,
            description="knowledge bank based on previous conversations",
        )
        return memory_tool

    def _init_long_term_memory_setter(self, embedding):
        memory_bank = PGMemoryStoreSetter(embedding, self.long_term_memory_collection)
        return memory_bank


if __name__ == "__main__":
    # test this class

    # select model and lora
    model_name = "llama-7b"
    lora_name = "alpaca-lora-7b"

    testAgent = LlamaModelHandler()
    eb = testAgent.get_hf_embedding()
    pipeline, model, tokenizer = testAgent.load_llama_llm(
        model_name=model_name, lora_name=lora_name, max_new_tokens=200
    )

    # define tool list (excluding any documents)
    test_tool_list = ["wiki", "searx"]

    # define test documents
    test_doc_info = {
        "examples": {
            "tool_name": "State of Union Document",
            "description": "President Joe Biden's 2023 state of the union address.",
            "files": ["index-docs/examples/state_of_the_union.txt"],
        }
    }

    # initiate agent executor
    kwarg = {
        "new_session": True,
        "use_cache_from_log": False,
        "log_tool_selector": False,
        "doc_use_type": "aggregate",
    }
    multi_step_agent = AgentMultiStepCritic(
        pipeline=pipeline,
        embedding=eb,
        tool_names=test_tool_list,
        doc_info=test_doc_info,
        verbose=True,
        **kwarg,
    )

    # testing start
    # print("testing for agent executor starts...")
    # test_prompt = "What did the president say about Ketanji Brown Jackson in his address to the nation?"
    # multi_step_agent.run(test_prompt)

    # # test with web ui
    # ui_run = WebUI(multi_step_agent.run)
    # ui_run.launch(server_name="0.0.0.0", server_port=7860)

    # finish
    print("testing complete")
