from collections import deque
from typing import Any, Deque, Dict, List, Optional, Sequence
from langchain.chains.llm import LLMChain
from pydantic import BaseModel, Field

from babyagi.task_creation_chain import TaskCreationChain
from babyagi.task_prioritization_chain import TaskPrioritizationChain
import babyagi.vectorstore as babyagi_vectorstore
from langchain.agents.agent import AgentExecutor

from langchain.vectorstores.base import VectorStore

from langchain.chains.base import Chain

from langchain.llms.base import BaseLLM

from langchain.tools.base import BaseTool

from langchain.agents.mrkl.base import ZeroShotAgent
from babyagi.tools import babyagi_tools


PREFIX = """You are an AI who performs one task based on the following objective: {objective}. Take into account these previously completed tasks: {context}."""
SUFFIX = """Question: {task}
{agent_scratchpad}"""

class BabyAGI(Chain, BaseModel):
    """Controller model for the BabyAGI agent."""

    task_list: Deque = Field(default_factory=deque)
    task_creation_chain: TaskCreationChain = Field(...)
    task_prioritization_chain: TaskPrioritizationChain = Field(...)
    execution_chain: AgentExecutor = Field(...)
    task_id_counter: int = Field(1)
    vectorstore: VectorStore = Field(init=False)
    max_iterations: Optional[int] = None
    class Config:
        """Configuration for this pydantic object."""

        arbitrary_types_allowed = True

    def add_task(self, task: Dict):
        self.task_list.append(task)

    def print_task_list(self):
        print("\033[95m\033[1m" + "\n*****TASK LIST*****\n" + "\033[0m\033[0m")
        for t in self.task_list:
            print(str(t["task_id"]) + ": " + t["task_name"])

    def print_next_task(self, task: Dict):
        print("\033[92m\033[1m" + "\n*****NEXT TASK*****\n" + "\033[0m\033[0m")
        print(str(task["task_id"]) + ": " + task["task_name"])

    def print_task_result(self, result: str):
        print("\033[93m\033[1m" + "\n*****TASK RESULT*****\n" + "\033[0m\033[0m")
        print(result)

    @property
    def input_keys(self) -> List[str]:
        return ["objective"]

    @property
    def output_keys(self) -> List[str]:
        return []
    
    def get_next_task(
      self,
      task_creation_chain: LLMChain,
      result: Dict,
      task_description: str,
      task_list: List[str],
      objective: str,
  ) -> List[Dict]:
      """Get the next task."""
      incomplete_tasks = ", ".join(task_list)
      response = task_creation_chain.run(
          result=result,
          task_description=task_description,
          incomplete_tasks=incomplete_tasks,
          objective=objective,
      )
      new_tasks = response.split("\n")
      return [{"task_name": task_name} for task_name in new_tasks if task_name.strip()]

    def prioritize_tasks(
            self,
        task_prioritization_chain: LLMChain,
        this_task_id: int,
        task_list: List[Dict],
        objective: str,
    ) -> List[Dict]:
        """Prioritize tasks."""
        task_names = [t["task_name"] for t in task_list]
        next_task_id = int(this_task_id) + 1
        response = task_prioritization_chain.run(
            task_names=task_names, next_task_id=next_task_id, objective=objective
        )
        new_tasks = response.split("\n")
        prioritized_task_list = []
        for task_string in new_tasks:
            if not task_string.strip():
                continue
            task_parts = task_string.strip().split(".", 1)
            if len(task_parts) == 2:
                task_id = task_parts[0].strip()
                task_name = task_parts[1].strip()
                prioritized_task_list.append({"task_id": task_id, "task_name": task_name})
        return prioritized_task_list


    def _get_top_tasks(self, vectorstore, query: str, k: int) -> List[str]:
        """Get the top k tasks based on the query."""
        results = vectorstore.similarity_search_with_score(query, k=k)
        if not results:
            return []
        sorted_results, _ = zip(*sorted(results, key=lambda x: x[1], reverse=True))
        return [str(item.metadata["task"]) for item in sorted_results]


    def execute_task(
        self, vectorstore, execution_chain: LLMChain, objective: str, task: str, k: int = 5
    ) -> str:
        """Execute a task."""
        context = self._get_top_tasks(vectorstore, query=objective, k=k)
        return execution_chain.run(objective=objective, context=context, task=task)

    def _call(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Run the agent."""
        objective = inputs["objective"]
        first_task = inputs.get("first_task", "Make a task list")
        self.add_task({"task_id": 1, "task_name": first_task})
        num_iters = 0
        while True:
            if self.task_list:
                self.print_task_list()

                # Step 1: Pull the first task
                task = self.task_list.popleft()
                self.print_next_task(task)

                # Step 2: Execute the task
                result = self.execute_task(
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
                new_tasks = self.get_next_task(
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
                    self.prioritize_tasks(
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
    def create_prompt(
        cls,
        tools: Sequence[BaseTool],
        prefix: str = PREFIX,
        suffix: str = SUFFIX,
    ):
        prompt = ZeroShotAgent.create_prompt(
            tools,
            prefix=prefix,
            suffix=suffix,
            input_variables=["objective", "task", "context", "agent_scratchpad"],
        )
        return prompt

    @classmethod
    def from_llm(
        cls,
        llm: BaseLLM,
        tools: Sequence[BaseTool] = babyagi_tools,
        verbose: bool = False,
        **kwargs
    ) -> "BabyAGI":
        """Initialize the BabyAGI Controller."""
        vectorstore = kwargs.get('vectorstore', babyagi_vectorstore.vectorstore)
        task_creation_chain = TaskCreationChain.from_llm(llm, verbose=verbose)
        task_prioritization_chain = TaskPrioritizationChain.from_llm(
            llm, verbose=verbose
        )
        prompt = cls.create_prompt(tools)
        llm_chain = LLMChain(llm=llm, prompt=prompt)
        tool_names = [tool.name for tool in tools]
        agent = ZeroShotAgent(llm_chain=llm_chain, allowed_tools=tool_names)
        agent_executor = AgentExecutor.from_agent_and_tools(
            agent=agent, tools=tools, verbose=False
        )
        return cls(
            task_creation_chain=task_creation_chain,
            task_prioritization_chain=task_prioritization_chain,
            execution_chain=agent_executor,
            vectorstore=vectorstore,
            **kwargs,
        )