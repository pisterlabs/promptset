from collections import deque
from typing import List
from langchain import LLMChain, OpenAI, PromptTemplate
from langchain.agents import ZeroShotAgent, AgentExecutor
import openai
from prompts import get_execution_prompt_template, templates
from tasks import Task
from tools import Tools


class Chain:
    def __init__(self, name: str):
        self.llm_chain = LLMChain(llm=OpenAI(temperature=0), prompt=self._get_prompt_template(name))
        self.name = name

    def _get_prompt_template(self, name: str):
        if name not in templates:
            raise ValueError(f"Invalid name: {name}")

        return PromptTemplate(**templates[name])
    
class TaskCreationChain:
    def __init__(self, chain_name: str):
        self.chain = Chain(chain_name)
    
    def create_task(self, objective: str, task: Task) -> Task:
        prompt = self.chain.llm_chain.prompt.get_prompt(
            objective=objective, context=[task.task_description]
        )
        task.task_description = self.chain.llm_chain.llm.generate(prompt=prompt, n=1)[0]
        return task


class TaskPriorityChain:
    def __init__(self, chain_name: str):
        self.chain = Chain(chain_name)

    def create_task_priorities(self, objective: str, tasks: List[Task]) -> List[Task]:
        prompt = self.chain.llm_chain.prompt.get_prompt(
            objective=objective, context=[task.task_description for task in tasks]
        )
        return self.chain.llm_chain.llm.generate(prompt=prompt, n=len(tasks))

class TaskExecutionChain:
    def __init__(self, chain_name: str, todo_chain: LLMChain):
        llm = OpenAI(temperature=0, max_tokens=256)
        self.tools = Tools(todo_chain=todo_chain, llm=llm)
        prompt = get_execution_prompt_template(self.tools)
        self.llm_chain = LLMChain(llm=llm, prompt=prompt)
        self.tool_names = [tool.name for tool in self.tools.tools]
        self.agent = ZeroShotAgent(llm_chain=self.llm_chain, allowed_tools=self.tool_names)
        self.agent_executor = AgentExecutor.from_agent_and_tools(
            agent=self.agent, tools=self.tools.tools, verbose=True)
        
    def get_ada_embedding(self, text: str):
        text = text.replace("\n", " ")
        return openai.Embedding.create(input=[text], model="text-embedding-ada-002")[
            "data"
        ][0]["embedding"]

    def context_agent(self, query: str, top_results_num: int, vector_index):
        query_embedding = self.get_ada_embedding(query)
        results = vector_index.query(query_embedding, top_k=top_results_num, include_metadata=True, namespace=query)
        sorted_results = sorted(results.matches, key=lambda x: x.score, reverse=True)
        return [(str(item.metadata["task"])) for item in sorted_results]

    def execute_task(self, objective: str, task: Task, index) -> str:
        context = self.context_agent(query=objective, top_results_num=5, vector_index=index)
        result = self.agent_executor.run(objective=objective, context=context, task=task.task_description)
        enriched_result = {
            "data": result
        }  # This is where you should enrich the result if needed
        result_id = f"result_{task.task_id}"
        vector = self.get_ada_embedding(
            enriched_result["data"]
        )  # get vector of the actual result extracted from the dictionary
        index.upsert(
            [(result_id, vector, {"task": task.task_name, "result": result})],
	    namespace=objective
        )

        return enriched_result

class TaskTodoChain:
    def __init__(self, chain_name: str):
        self.task_list = deque()
        self.chain = Chain(chain_name)
