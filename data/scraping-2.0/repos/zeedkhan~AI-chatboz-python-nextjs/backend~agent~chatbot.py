import os
from dotenv import load_dotenv
from typing import List,  Dict, TypedDict
from datetime import datetime
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate
from langchain.chat_models import ChatOpenAI
import asyncio
from langchain import PromptTemplate
from backend.agent.take_output_parser import TaskOutputParser, ConvoTaskOutputParser, NormalOutputParser
from backend.agent.helper import parse_with_handling, call_model_with_handling
from backend.agent.analysis import Analysis, AnalysisArguments
from backend.agent.prompt import analyze_task_prompt, conversation_prompt, answer_prompt
from backend.agent.helper import openai_error_handler
from langchain.output_parsers import PydanticOutputParser
from backend.agent.open_ai_functions import get_tool_function
from backend.agent.utils.tool_fn import get_tool_from_name, get_tool_name, get_default_tool
from backend.agent.utils.tool_fn import get_available_tools, get_available_tools_names, get_user_tools
from starlette.routing import Route
from starlette.authentication import requires
from starlette.responses import JSONResponse
from starlette.requests import Request


load_dotenv()

open_ai_key = os.getenv("OPENAI_API_KEY")


start_goal_prompt = PromptTemplate(
    template="""
        You are a task creation AI. Your responses should be in the "{language}" language.  
        Your process involves understanding problems, extracting variables, and devising complete plans.

        Objective: "{goal}"

        Your goal is to create a step-by-step plan to achieve the given objective.

        You possess high-priority action and reasoning that are most relevant to the goal: [{priority_functions}]. 
        Prioritize the use of these functions and consider available functions as additional options when needed.

        Take into account the available functions and their parameters if required. The available functions are: {tool_list}.

        Return the steps with as a formatted array of strings that can be used with JSON.parse().
        Include step not a guideline to the array!.
        Do not include such as Create, Edit if these command did not come from the goal!.
        
        Examples:
        0. ["Retrieve Google Cloud Connection using GoogleCloudProject", "Retrieve Google Cloud instances using GoogleCloudInstances"]
        1. ["Search the web for NBA news to get informations about Stephen Curry.", "Compile a financial report on Nike."]
        2. ["Develop a function to add a new vertex with a specified weight to the digraph."]
        3. ["Gather further information about Bertie W.", "Explore and document the ultimate Kentucky Fried Chicken recipe."]
    """,
    input_variables=["goal", "language", "tool_list", "priority_functions"],
)


class CustomAgent():
    def __init__(
        self, model
    ):
        self.model = model

    async def chat_bot(
        self,
        goal: str = "",
        lang: str = "",
        user: str = "",
        ai_prefix: str = ""
    ) -> Dict:

        task_output_parser = ConvoTaskOutputParser()
        tool_names = get_available_tools_names()

        tools = list(map(get_tool_function, get_user_tools(tool_names)))

        completion = await call_model_with_handling(
            self.model,
            ChatPromptTemplate.from_messages(
                [SystemMessagePromptTemplate(prompt=conversation_prompt)]
            ),
            {
                "question": goal,
                "language": lang,
                "user": user,
                "ai_prefix": ai_prefix,
                "functions": tools
            }
        )

        tasks = parse_with_handling(task_output_parser, completion)

        return tasks

    async def plan_agent(self, goal: str = "", lang: str = "", priority_functions: List[str] = []) -> List[str]:

        task_output_parser = TaskOutputParser(completed_tasks=[])

        tools = get_available_tools()

        completion = await call_model_with_handling(
            self.model,
            ChatPromptTemplate.from_messages(
                [SystemMessagePromptTemplate(prompt=start_goal_prompt)]
            ),
            {
                "goal": goal,
                "language": lang,
                "tool_list": tools,
                "priority_functions": priority_functions
            },
        )

        tasks = parse_with_handling(task_output_parser, completion)

        return tasks

    async def analyze_task_agent(
        self,
        goal: str,
        task: str,
        tool_names: List[str] = get_available_tools_names(),
        all_values: List[dict] = []
    ) -> Analysis:
        functions = list(map(get_tool_function, get_user_tools(tool_names)))

        prompt = analyze_task_prompt.format_prompt(
            goal=goal,
            task=task,
            language="English",
            all_values=all_values
        )

        message = await openai_error_handler(
            func=self.model.apredict_messages,
            messages=prompt.to_messages(),
            functions=functions,
        )

        function_call = message.additional_kwargs.get("function_call", {})
        completion = function_call.get("arguments", "")

        try:
            pydantic_parser = PydanticOutputParser(
                pydantic_object=AnalysisArguments)
            analysis_arguments = parse_with_handling(
                pydantic_parser, completion)
            return Analysis(
                action=function_call.get(
                    "name", get_tool_name(get_default_tool())),
                **analysis_arguments.dict()
            )
        except ValueError as e:
            print("Error")
            print(e)
            return Analysis.get_default_analysis()

    async def execute_task_agent(
        self,
        goal: str,
        task: str,
        analysis: Analysis,
    ):
        # TODO: More mature way of calculating max_tokens
        # functions = [tool["name"] for tool in list(map(get_tool_function, get_user_tools(tool_names)))]

        tool_class = get_tool_from_name(analysis.action)
        return await tool_class(self.model, language="English").call(
            goal, task, analysis.arg
        )

    async def answer_agent(
        self,
        goal: str,
        plans: List[str],
        all_values: List[dict]
    ):

        task_output_parser = NormalOutputParser()

        completion = await call_model_with_handling(
            self.model,
            ChatPromptTemplate.from_messages(
                [SystemMessagePromptTemplate(prompt=answer_prompt)]
            ),
            {
                "goal": goal,
                "plans": plans,
                "language": "English",
                "all_values": all_values
            },
        )

        answer = parse_with_handling(task_output_parser, completion)

        return answer


agent = CustomAgent(model=ChatOpenAI(model="gpt-3.5-turbo"))


async def chat(question="", language="", ai_prefix="", user=""):
    conversation = await agent.chat_bot(goal=question, lang=language, ai_prefix=ai_prefix, user=user)

    return conversation


async def plan(goal="", lang="", priority_functions: List[str] = []):
    plan = await agent.plan_agent(goal, lang, priority_functions)

    return plan


async def anlysis(goal, task, all_values):
    plan = await agent.analyze_task_agent(goal, task, all_values=all_values)

    return plan


async def exec(goal, task, analysis):
    exe = await agent.execute_task_agent(goal, task, analysis)
    return exe


async def answer(goal, plans, all_values):
    answer = await agent.answer_agent(goal, plans, all_values)

    return {
        "response": answer
    }


def tools(request):
    user = request.user
    print(request)
    if user.is_authenticated:
        tool_names = get_available_tools_names()
        functions = list(map(get_tool_function, get_user_tools(tool_names)))
        response = JSONResponse({
            "tools": functions
        })
        response.status_code = 200

        print(functions)

        return response

    return user.is_authenticated


def start(goal):
    if not goal:
        goal = "Do you know my girlfriend?"

    lang = "English"

    while True:

        conver = asyncio.run(
            chat(question=goal, language=lang, ai_prefix="Seed Junior", user="Seed"))

        if conver["use_function"]:

            priority_functions = conver["response"]

            plans = asyncio.run(
                plan(goal=goal, lang=lang, priority_functions=priority_functions))
            print("*" * 50)
            print("plans")
            print(plans)

            all_values = []

            while plans:
                current_task = plans.pop(0)
                thinking = asyncio.run(
                    anlysis(goal=goal, task=current_task, all_values=all_values))
                print("*" * 50)
                print("task")
                print(current_task)
                print("*" * 50)
                print("thinking")
                print(thinking)
                exe = asyncio.run(
                    exec(goal=goal, task=current_task, analysis=thinking))
                all_values.append({"task": current_task, "value": exe})
                print("*" * 50)
                print("return")
                print(exe)
                print("*" * 50)

            asnwer_question = asyncio.run(
                answer(goal=goal, plans=plans, all_values=all_values))

            return asnwer_question

        return conver


async def test(request: Request):
    try:
        data = await request.json()
        print(data)
        # Do something with the POST data
        result = {"message": "Received POST data successfully", "data": data}
        return JSONResponse(content=result, status_code=200)
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)


class ChatInit(TypedDict):
    goal: str
    language: str
    ai_prefix: str
    user: str


async def test_chat(request: Request):
    data: ChatInit = await request.json()
    question = data.get("goal")
    lang = data.get("language")
    ai_prefix = data.get("ai_prefix")
    user = data.get("user")

    conversation = await agent.chat_bot(goal=question, lang=lang, ai_prefix=ai_prefix, user=user)

    return JSONResponse({"data": conversation}, status_code=200)


chat_router = [
    Route('/tools/', endpoint=requires('authenticated')(tools)),
    Route('/test/', endpoint=requires('authenticated')
          (test), methods=['POST']),
    Route('/chat/', endpoint=requires('authenticated')
          (test_chat), methods=['POST']),
]


# test = start()

# print(test)
