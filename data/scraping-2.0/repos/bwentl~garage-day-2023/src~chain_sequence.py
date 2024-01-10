import sys
import re
from typing import Any, List, Optional, Type, Dict

from langchain.chains import LLMChain
from langchain.chains import RetrievalQA
from langchain.chains.constitutional_ai.base import ConstitutionalChain
from langchain.chains.constitutional_ai.models import ConstitutionalPrinciple
from langchain.prompts.prompt import PromptTemplate

sys.path.append("./")
from src.models import LlamaModelHandler
from src.docs import DocumentHandler
from src.tools import ToolHandler
from src.util import get_secrets, get_word_match_list, agent_logs
from src.prompts.examples import (
    CHAIN_EXAMPLE_A1,
    CHAIN_EXAMPLE_A2,
    CHAIN_EXAMPLE_A3,
    CHAIN_EXAMPLE_B1,
    CHAIN_EXAMPLE_B2,
    CHAIN_EXAMPLE_B3,
)


class ChainSequence:
    """a simple wrapper around different LLM Chain types to perform complex task"""

    # https://python.langchain.com/en/latest/modules/chains/getting_started.html#create-a-custom-chain-with-the-chain-class

    def __init__(self, config, pipeline, **kwarg):
        """
        example for chains:
        chain_config = [
            {
                "name": "task1",
                "type": "simple",
                "input_template": "Give me one name for a company that makes {input}?",
            },
            {
                "name": "task2",
                "type": "simple",
                "input_template": "What is a good slogan for a company that makes {input} and named {task1_output}?",
            },
        ]
        """
        self.new_session = kwarg["new_session"] if "new_session" in kwarg else False
        self.use_cache_from_log = (
            kwarg["use_cache_from_log"] if "use_cache_from_log" in kwarg else False
        )
        self.chain_name = (
            kwarg["chain_name"] if "chain_name" in kwarg else "custom chain"
        )

        self.chains = dict()
        self.outputs = {"input": ""}
        # TODO: add support for chain serialization - https://python.langchain.com/en/latest/modules/chains/generic/serialization.html
        # notes on different chain types
        # https://python.langchain.com/en/latest/modules/chains/index_examples/summarize.html

        for c in config:
            task_name = c["name"]
            self.chains[task_name] = c
            if c["type"] == "simple":
                self.chains[task_name]["chain"] = self._init_llm_chain(c, pipeline)
            elif c["type"] == "constitutional":
                base_chain = self._init_llm_chain(c, pipeline)
                self.chains[task_name]["chain"] = self._init_constitutional_chain(
                    llm_chain=base_chain, llm=pipeline, principles=c["principles"]
                )
            else:
                raise ValueError("chain type not currently supported.")

    def run(self, input):
        # set task list
        task_list = list(self.chains.keys())
        task_list_str = "+".join(task_list)

        if self.new_session:
            # set cache state to save cache logs
            cached_response = agent_logs.set_cache_lookup(
                f"Custom Chains - {task_list_str} - {input}"
            )
            # if using cache from logs saved, then try to load previous log
            if cached_response is not None and self.use_cache_from_log:
                return cached_response
            agent_logs().clear_log()

        print(f"> Initiating {self.chain_name} sequence for {task_list_str}...")
        self.outputs["input"] = input
        for task_name in task_list:
            c = self.chains[task_name]
            # run extra tool action if type is qa_with_tool
            if "tool" in c and "tool_input" in c:
                self.outputs[f"{task_name}_summaries"] = c["tool"](
                    self.outputs[c["tool_input"]]
                )
            values = [self.outputs[v] for v in c["input_vars"]]
            input_list = {key: value for key, value in zip(c["input_vars"], values)}
            raw_output = c["chain"].apply([input_list])
            if c["type"] == "constitutional":
                current_output = raw_output[0]["output"].replace("Model: ", "")
            else:
                current_output = raw_output[0]["text"].strip()

            self.outputs[f"{task_name}_output"] = current_output
            self.outputs["final_output"] = current_output
            if task_name is not task_list[-1]:
                agent_logs.write_log_and_print(current_output, ans_type="answer")
        agent_logs.write_log_and_print(self.outputs["final_output"], ans_type="final")

        # always cache the current log
        agent_logs.save_cache()

        return self.outputs["final_output"]

    def _init_llm_chain(self, config, llm):
        template = config["input_template"]
        inputs = re.findall(r"\{(\w+)\}", template)
        config["input_vars"] = inputs
        prompt_template = PromptTemplate(
            input_variables=inputs,
            template=template,
        )
        chain = LLMChain(llm=llm, prompt=prompt_template, verbose=True)
        return chain

    def _init_constitutional_chain(self, llm_chain, llm, principles=["ethical"]):
        constitutional_principles = List()
        for pre_defined in principles:
            if isinstance(pre_defined, ConstitutionalPrinciple):
                constitutional_principles.append(constitutional_principles)
            else:
                if pre_defined == "ethical":
                    principle = ConstitutionalPrinciple(
                        name="Ethical Principle",
                        critique_request="The model should only talk about ethical and legal things.",
                        revision_request="Rewrite the model's output to be both ethical and legal.",
                    )
                    constitutional_principles.append(principle)
                elif pre_defined == "thoughtful":
                    principle = ConstitutionalPrinciple(
                        name="Thoughtful Principle",
                        critique_request="The model should provide response that is relevant to the initial question.",
                        revision_request="Rewrite the model's output to answer the initial question with all the information provided.",
                    )
                    constitutional_principles.append(principle)
                else:
                    raise NotImplementedError(
                        f"{pre_defined} is not supported as one of the constitutional principles."
                    )

        constitutional_chain = ConstitutionalChain.from_llm(
            chain=llm_chain,
            constitutional_principles=constitutional_principles,
            llm=llm,
            verbose=True,
        )
        return constitutional_chain


if __name__ == "__main__":
    # test this class

    # select model and lora
    model_name = "llama-7b"
    lora_name = "alpaca-lora-7b"

    testAgent = LlamaModelHandler()
    embedding = testAgent.get_hf_embedding()
    pipeline, model, tokenizer = testAgent.load_llama_llm(
        model_name=model_name, lora_name=lora_name, max_new_tokens=200
    )

    # test 1

    # get search tool
    tools_wrapper = ToolHandler()
    searx_tool = tools_wrapper.get_tools(["searx"], pipeline)[0]
    # get document tool
    test_doc_info = {
        "examples": {
            "tool_name": "State of Union Document",
            "description": "President Joe Biden's 2023 state of the union address.",
            "files": ["index-docs/examples/state_of_the_union.txt"],
        },
        "memories": {
            "tool_name": "Conversation History",
            "description": "history of previous conversations by the AI agent.",
            "memory_type": "long_term",
        },
    }
    # add document retrievers to tools
    if len(test_doc_info) > 0:
        newDocs = DocumentHandler(
            embedding=embedding, redis_host=get_secrets("redis_host")
        )
        tools_list = newDocs.get_tool_from_doc(
            pipeline=pipeline,
            doc_info=test_doc_info,
            doc_use_type="aggregate",
            doc_top_k_results=3,
        )
        example_doc_tool = tools_list[0]
        memory_doc_tool = tools_list[1]
    # build chain config
    chain_config = [
        {
            "name": "task1",
            "type": "simple",
            "input_template": CHAIN_EXAMPLE_B1,
            "tool": memory_doc_tool,
            "tool_input": "input",
        },
        {
            "name": "task2",
            "type": "simple",
            "input_template": CHAIN_EXAMPLE_B2,
        },
        {
            "name": "task3",
            "type": "constitutional",
            "principles": ["ethical"],
            "input_template": CHAIN_EXAMPLE_B3,
            "tool": searx_tool,
            "tool_input": "task2_output",
        },
    ]

    # start ui
    from src.gradio_ui import WebUI

    # run chains
    custom_chains = ChainSequence(config=chain_config, pipeline=pipeline)
    # custom_chains.run(input="What happened yesterday? Any big news?")

    ui_run = WebUI(custom_chains.run)
    ui_run.launch(server_name="0.0.0.0", server_port=7860)

    # # test 2
    # args = {
    #     "use_cache_from_log": True,
    # }
    # chain_config = [
    #     {
    #         "name": "task1",
    #         "type": "simple",
    #         "input_template": CHAIN_EXAMPLE_A1,
    #     },
    #     {
    #         "name": "task2",
    #         "type": "simple",
    #         "input_template": CHAIN_EXAMPLE_A2,
    #     },
    #     {
    #         "name": "task3",
    #         "type": "simple",
    #         "input_template": CHAIN_EXAMPLE_A3,
    #     },
    # ]
    # # run chains
    # custom_chains = ChainSequence(config=chain_config, pipeline=pipeline, **args)
    # # custom_chains.run(input="what did the president say about Ketanji Brown Jackson")

    print("test done")
