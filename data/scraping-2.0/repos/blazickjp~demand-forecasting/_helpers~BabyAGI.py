import json
import os
from typing import Any, Dict, List, Optional

import faiss
from langchain import GoogleSearchAPIWrapper, LLMChain, OpenAI, PromptTemplate
from langchain.agents import AgentExecutor, Tool, ZeroShotAgent
from langchain.chains.base import Chain
from langchain.docstore import InMemoryDocstore
from langchain.embeddings import OpenAIEmbeddings
from langchain.llms import BaseLLM
from langchain.vectorstores import FAISS
from langchain.vectorstores.base import VectorStore
from pydantic import BaseModel, Field

# Use a service account
# cred = credentials.Certificate('cfa-creds.json')
# firebase_admin.initialize_app(cred)

# db = firestore.client()

# Define your embedding model
embeddings_model = OpenAIEmbeddings()

# Initialize the vectorstore as empty
embedding_size = 1536
index = faiss.IndexFlatL2(embedding_size)
vectorstore = FAISS(embeddings_model.embed_query,
                    index, InMemoryDocstore({}), {})


class TaskProgressChain(LLMChain):
    """Chain to manage the progress of tasks."""

    @classmethod
    def from_llm(cls, llm: BaseLLM, verbose: bool = True) -> LLMChain:
        """Get the response parser."""
        task_progress_template = (
            "You are a task progress AI that uses the current task and its metadata"
            " to manage the progress of the task. The current task is: {task_name},"
            " and its metadata is: {metadata}. Based on this information,"
            " provide a summary of the progress on task completion."
        )
        prompt = PromptTemplate(
            template=task_progress_template,
            input_variables=["task_name", "metadata"],
        )
        return cls(prompt=prompt, llm=llm, verbose=verbose)


class TaskCreationChain(LLMChain):
    """_summary_

    Args:
        LLMChain (_type_): _description_

    Returns:
        _type_: _description_
    """

    @classmethod
    def from_llm(cls, llm: BaseLLM, verbose: bool = True) -> LLMChain:
        """Get the response parser."""
        task_creation_template = (
            "You are an task creation AI that uses the result of an execution agent"
            " to create new tasks with the following objective: {objective},"
            " The list of completed tasks are: {completed_tasks}."
            " These are the incomplete tasks still remaining: {incomplete_tasks}."
            " Based on the result, create new tasks when necessary."
            " Return the tasks as an array."
        )
        prompt = PromptTemplate(
            template=task_creation_template,
            input_variables=["completed_tasks", "incomplete_tasks", "objective"],
        )
        return cls(prompt=prompt, llm=llm, verbose=verbose)


class TaskPrioritizationChain(LLMChain):
    """_summary_

    Args:
        LLMChain (_type_): _description_

    Returns:
        _type_: _description_
    """
    @classmethod
    def from_llm(cls, llm: BaseLLM, verbose: bool = True) -> LLMChain:
        """Get the response parser."""
        task_prioritization_template = (
            "You are an task prioritization AI tasked with cleaning, formatting, and reprioritizing"
            " the following tasks: {task_names}."
            " Consider the ultimate objective of your team: {objective}."
            " Do not remove any tasks. Return the result as a numbered list, like:"
            " #. First task"
            " #. Second task"
            " Start the task list with number {next_task_id}."
        )
        prompt = PromptTemplate(
            template=task_prioritization_template,
            input_variables=["task_names", "next_task_id", "objective"],
        )
        return cls(prompt=prompt, llm=llm, verbose=verbose)


todo_prompt = PromptTemplate.from_template(
    "You are a planner who is an expert at coming up with a todo list for a given objective. Come up with a todo list for this objective: {objective}")
todo_chain = LLMChain(llm=OpenAI(temperature=0), prompt=todo_prompt)

search = GoogleSearchAPIWrapper()


def search_parser(string: str):
    """Parse the search results."""
    return search.results(string, num_results=4)


# Create a prompt template for text summarization
summarize_prompt = PromptTemplate.from_template("Please summarize the following text: {text}")

# Create an LLMChain for text summarization using the OpenAI language model and the prompt template
summarize_chain = LLMChain(llm=OpenAI(temperature=0.3), prompt=summarize_prompt)


def save_summary_to_file(text: str, file_path: str) -> str:
    """Summarize the given text and save the summary to a file.

    Args:
        text: The text to be summarized.
        file_path: The path to the file where the summary will be saved.

    Returns:
        A string indicating the path to the file where the summary was saved.
    """
    # Use the summarize_chain to generate a summary of the text
    summary = summarize_chain.run({"text": text})

    # Save the summary to a file
    with open(file_path, 'w') as f:
        f.write(summary)

    return f"Summary saved to {file_path}"


# Define the file path where the summary will be saved
file_path = os.path.join(os.getcwd(), 'summary.txt')

# Add a tool to your tools list that summarizes text and saves the summary to a file
tools = []
tools = [
    Tool(
        name="Summarize and Save",
        func=save_summary_to_file,
        description=f"useful for when you need to summarize a large amount of text and save the summary to a file. Input: a large amount of text. Output: a string indicating the path to the file where the summary was saved. The summary will be saved to {file_path}."
    ),
    Tool(
        name="Search",
        func=search_parser,
        description="useful for when you need to gather real time information using Google Search. Input: a search query. Output: the top 4 search results."
    ),
]


prefix = """
You are an AI who performs one task keeping in mind that your final objective is: {objective}.
Take into account these previously completed tasks: {context}
Also take into account a summary of your current progress: {summary}
"""
FORMAT_INSTRUCTIONS = """Use the following format:
Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question"""
suffix = """Question: {task}
{agent_scratchpad}"""
prompt = ZeroShotAgent.create_prompt(
    tools,
    prefix=prefix,
    format_instructions=FORMAT_INSTRUCTIONS,
    suffix=suffix,
    input_variables=["objective", "task", "context", "summary", "agent_scratchpad"]
)


class BabyAGI(Chain, BaseModel):
    """Controller model for the BabyAGI agent."""
    task_creation_chain: TaskCreationChain = Field(...)
    task_prioritization_chain: TaskPrioritizationChain = Field(...)
    task_progress_chain: TaskProgressChain = Field(...)
    todo_chain: LLMChain = Field(...)
    execution_chain: AgentExecutor = Field(...)
    task_id_counter: int = Field(0)
    task_list: List = Field([])
    completed_tasks: List = Field([])
    vectorstore: VectorStore = Field(init=False)
    objective: str = Field("")
    max_iterations: Optional[int] = None
    task_file_path: str = Field("task_list.json")

    class Config:
        """Configuration for this pydantic object."""
        arbitrary_types_allowed = True

    @property
    def input_keys(self) -> List[str]:
        return ["objective"]

    @property
    def output_keys(self) -> List[str]:
        return []

    def initialize_task_file(self):
        if os.path.exists(self.task_file_path):
            os.remove(self.task_file_path)

    def get_task(self, task_id: str):
        return [t for t in self.task_list if t["task_id"] == str(task_id)]

    def get_max_task_id(self):
        return max([int(t["task_id"]) for t in self.task_list])

    def add_task(self, task):
        self.task_list.append(task)

    def initialize_task_list(self):
        """
        Initialize the task list with a TODO task.
        """
        self.add_task({"task_id": "0", "task_name": "Make a Todo List", "iteration": 0,
                       "status": "incomplete", "summary": ""})
        self.refresh_task_list()
        self.save_task_list()

    def execute_and_update_task(self, task):
        response = self.execution_chain(
            inputs={
                "objective": self.objective,
                "context": self.completed_tasks,
                "task": task,
                "summary": task["summary"]
            }
        )
        result = response["output"]
        intermediate_steps = response["intermediate_steps"]
        task.update({f"result_{task['iteration']}": result})
        task.update({f"step_{task['iteration']}": intermediate_steps})
        task.update({"iteration": str(int(task["iteration"]) + 1)})

        self.save_task_list()

    def reprioritize_tasks(self):
        starting_task_id = self.task_id_counter + 1
        task_names = [i for i in self.get_task_names(starting_task_id)]
        reprioritized_tasks = self.task_prioritization_chain(
            inputs={"objective": self.objective, "next_task_id": starting_task_id, "task_names": task_names}
        )
        reprioritized_task_list = reprioritized_tasks["text"].split("\n")
        task = self.get_task(str(self.task_id_counter))
        for task_name in reprioritized_task_list:
            if task_name.strip():
                task.append({"task_id": str(starting_task_id), "task_name": task_name,
                             "iteration": 0, "status": "incomplete", "summary": ""})
                starting_task_id += 1

        self.task_list = task
        self.save_task_list()

    def save_task_list(self):
        with open(self.task_file_path, 'w') as file:
            json.dump(self.task_list, file)

    def load_task_list(self):
        with open(self.task_file_path, 'r') as file:
            self.task_list = json.load(file)

    def get_task_names(self, starting_task_id: int):
        return [t["task_name"] for t in self.task_list if int(t["task_id"]) >= starting_task_id]

    def remove_task(self, task):
        new_tasks = [t for t in self.task_list if t["task_id"] != task["task_id"]]
        self.task_list = new_tasks
        self.save_task_list()

    def refresh_task_list(self):
        """Get the next task."""
        max_id = self.get_max_task_id()
        response = todo_chain.run(
            incomplete_tasks=self.task_list, objective=self.objective, completed_tasks=self.completed_tasks)
        new_tasks = response.split('\n')
        new_task_list = [{"task_name": task_name, "iteration": "0", "status": "incomplete", "summary": "", "task_id": j + max_id}
                         for j, task_name in enumerate(new_tasks) if task_name.strip()]
        for i in new_task_list:
            self.add_task(i)
        self.save_task_list()

    def _call(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Run the agent."""
        self.objective = inputs['objective']
        self.initialize_task_file()
        self.initialize_task_list()
        while True:
            if self.task_list:
                input("Check Progress and update Status. Press Enter to continue...")
                self.load_task_list()
                task = self.get_task(str(self.task_id_counter))[0]
                if task['status'] == "complete":
                    print("Completed task!\t- ", task['task_name'])
                    self.completed_tasks.append(task)
                    self.remove_task(task)
                    self.reprioritize_tasks()
                    self.task_id_counter += 1
                else:
                    self.execute_and_update_task(task)
                    self.save_task_list()
            else:
                print("No Tasks left to do!")
                break
        return {}

    @classmethod
    def from_llm(
        cls,
        llm: BaseLLM,
        vectorstore: VectorStore,
        verbose: bool = False,
        **kwargs
    ) -> "BabyAGI":
        """Initialize the BabyAGI Controller."""
        task_creation_chain = TaskCreationChain.from_llm(
            llm, verbose=verbose
        )
        task_prioritization_chain = TaskPrioritizationChain.from_llm(
            llm, verbose=verbose
        )
        task_progress_chain = TaskProgressChain.from_llm(
            llm, verbose=verbose
        )
        llm_chain = LLMChain(llm=llm, prompt=prompt)
        tool_names = [tool.name for tool in tools]
        agent = ZeroShotAgent(llm_chain=llm_chain, allowed_tools=tool_names)
        agent_executor = AgentExecutor.from_agent_and_tools(
            agent=agent, tools=tools, verbose=True, return_intermediate_steps=True)  # Change this line
        return cls(
            task_creation_chain=task_creation_chain,
            task_prioritization_chain=task_prioritization_chain,
            execution_chain=agent_executor,
            task_progress_chain=task_progress_chain,
            vectorstore=vectorstore,
            todo_chain=todo_chain,
            **kwargs
        )


OBJECTIVE = "Research the Langchain Python documentation and generate some product ideas leveraging the python package"

llm = OpenAI(temperature=0)
# Logging of LLMChains
verbose = False
# If None, will keep on going forever
max_iterations: Optional[int] = 7
baby_agi = BabyAGI.from_llm(
    llm=llm,
    vectorstore=vectorstore,
    verbose=verbose,
    max_iterations=max_iterations
)

baby_agi({"objective": OBJECTIVE})
