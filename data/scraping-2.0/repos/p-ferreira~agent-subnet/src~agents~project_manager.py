from .agent import BaseAgent, ProjectPlan, Task
from loguru import logger
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.utils.openai_functions import convert_pydantic_to_openai_function
from langchain.output_parsers.openai_functions import JsonOutputFunctionsParser    


def run_project_planning_chain(model: ChatOpenAI, requirements: str) -> str:
    project_plan_template = """
You are a professional project manager; the goal is to break down the list of requirements in triple backticks into tasks to be given to a software engineer.
Your tasks should contain:
- task description: The work that the programmer will need to perform
- acceptance criteria: What is the acceptance criteria to define the success of the task to be performed
"""
    prompt = ChatPromptTemplate.from_messages([
        ("system", project_plan_template),
        ("user", "{input}")
    ])

    # Run the project planning chain
    project_plan_generation_chain = prompt | model
    project_plan = project_plan_generation_chain.invoke({"input": requirements})    

    return project_plan.content


def run_project_creation_chain(model: ChatOpenAI, project_plan: str) -> str:
    task_extraction_functions = [convert_pydantic_to_openai_function(ProjectPlan)]
    task_extraction_model = model.bind(functions=task_extraction_functions, function_call={"name" : "ProjectPlan"})

    task_organization_prompt = ChatPromptTemplate.from_messages([
        ("system", "Extract the project plan alogside the tasks structured in task description and acceptance criteria, do not guess or invent information."),
        ("human", "{project_plan}")
    ])

    task_organization_chain = task_organization_prompt | task_extraction_model | JsonOutputFunctionsParser(keyname="ProjectPlan")

    raw_project_plan = task_organization_chain.invoke({"project_plan": project_plan})
    raw_project_plan['plan'] = project_plan

    mapped_project_plan = ProjectPlan(**raw_project_plan)
    return mapped_project_plan


class ProjectManagerAgent(BaseAgent):
    def __init__(self):
        super().__init__()

    
    def create_project_plan(self, requirements: str) -> ProjectPlan:
        logger.info("Creating initial draft for project plan...")
        initial_project_planning: str= run_project_planning_chain(self.model, requirements)

        logger.info("Mapping project plan...")
        project_plan: ProjectPlan = run_project_creation_chain(self.model, initial_project_planning)

        return project_plan