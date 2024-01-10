import argparse
import asyncio
import inspect
import json
import os
import sys
import traceback
from threading import Thread

import openai
from openai import OpenAI, AsyncOpenAI, NotFoundError
from openai.types.chat.chat_completion_message import FunctionCall

import baserun


@baserun.trace
def openai_chat(prompt="What is the capital of the US?") -> str:
    client = OpenAI()
    completion = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
    )
    content = completion.choices[0].message.content
    baserun.check_includes("openai_chat.content", content, "Washington")
    return content


@baserun.trace
def openai_chat_with_log(prompt="What is the capital of the US?") -> str:
    client = OpenAI()
    completion = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
    )
    content = completion.choices[0].message.content
    baserun.check_includes("openai_chat.content", content, "Washington")
    command = " ".join(sys.argv)
    baserun.log(f"OpenAI Chat Results", payload={"command": command})
    return content


def openai_chat_unwrapped(prompt="What is the capital of the US?", **kwargs) -> str:
    client = OpenAI()
    completion = client.chat.completions.create(
        model="gpt-3.5-turbo", messages=[{"role": "user", "content": prompt}], **kwargs
    )
    return completion.choices[0].message.content


async def openai_chat_unwrapped_async_streaming(prompt="What is the capital of the US?", **kwargs) -> str:
    client = AsyncOpenAI()
    completion_generator = await client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        **kwargs,
        stream=True,
    )
    content = ""
    async for chunk in completion_generator:
        if new_content := chunk.choices[0].delta.content:
            content += new_content

    return content


@baserun.trace
async def openai_chat_async(prompt="What is the capital of the US?") -> str:
    client = AsyncOpenAI()
    completion = await client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
    )
    content = completion.choices[0].message.content
    baserun.check_includes("openai_chat_async.content", content, "Washington")
    return content


@baserun.trace
def openai_chat_tools(prompt="Say 'hello world'") -> FunctionCall:
    client = OpenAI()
    tools = [
        {
            "type": "function",
            "function": {
                "name": "say",
                "description": "Convert some text to speech",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "text": {"type": "string", "description": "The text to speak"},
                    },
                    "required": ["text"],
                },
            },
        }
    ]
    completion = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        tools=tools,
        tool_choice={"type": "function", "function": {"name": "say"}},
    )
    tool_calls = completion.choices[0].message.tool_calls
    baserun.check_includes(
        "openai_chat_functions.function_call",
        json.dumps([{"id": call.id, "type": call.type, "function": call.function.__dict__} for call in tool_calls]),
        "say",
    )
    return tool_calls


@baserun.trace
def openai_chat_functions(prompt="Say 'hello world'") -> FunctionCall:
    client = OpenAI()
    functions = [
        {
            "name": "say",
            "description": "Convert some text to speech",
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {"type": "string", "description": "The text to speak"},
                },
                "required": ["text"],
            },
        }
    ]
    completion = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        functions=functions,
        function_call={"name": "say"},
    )
    fn_call = completion.choices[0].message.function_call
    baserun.check_includes("openai_chat_functions.function_call", json.dumps(fn_call.__dict__), "say")
    return fn_call


@baserun.trace
def openai_chat_functions_streaming(prompt="Say 'hello world'") -> FunctionCall:
    client = OpenAI()
    functions = [
        {
            "name": "say",
            "description": "Convert some text to speech",
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {"type": "string", "description": "The text to speak"},
                },
                "required": ["text"],
            },
        }
    ]
    completion_generator = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        functions=functions,
        stream=True,
        function_call={"name": "say"},
    )
    function_name = ""
    function_arguments = ""
    for chunk in completion_generator:
        choice = chunk.choices[0]
        if function_call := choice.delta.function_call:
            if function_call.name:
                function_name += function_call.name

            if function_call.arguments:
                function_arguments += function_call.arguments

    baserun.check_includes("openai_chat_functions.function_call_streaming", function_name, "say")
    return {"name": function_name, "arguments": function_arguments}


@baserun.trace
def openai_chat_streaming(prompt="What is the capital of the US?") -> str:
    client = OpenAI()
    completion_generator = client.chat.completions.create(
        model="gpt-3.5-turbo",
        stream=True,
        messages=[{"role": "user", "content": prompt}],
    )
    content = ""
    for chunk in completion_generator:
        if new_content := chunk.choices[0].delta.content:
            content += new_content

    baserun.check_includes("openai_chat_streaming.content", content, "Washington")
    return content


@baserun.trace
async def openai_chat_async_streaming(prompt="What is the capital of the US?") -> str:
    client = AsyncOpenAI()
    completion_generator = await client.chat.completions.create(
        model="gpt-3.5-turbo",
        stream=True,
        messages=[{"role": "user", "content": prompt}],
    )
    content = ""
    async for chunk in completion_generator:
        if new_content := chunk.choices[0].delta.content:
            content += new_content

    baserun.check_includes("openai_chat_async_streaming.content", content, "Washington")
    return content


@baserun.trace
def openai_chat_error(prompt="What is the capital of the US?"):
    client = OpenAI()

    original_api_type = openai.api_type
    try:
        client.chat.completions.create(
            model="asdf",
            messages=[{"role": "user", "content": prompt}],
        )
    except NotFoundError as e:
        baserun.check_includes("openai_chat_async_streaming.content", e.message, "does not exist")
        raise e
    finally:
        openai.api_type = original_api_type


@baserun.trace
def traced_fn_error():
    client = OpenAI()

    client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": "What is the capital of the US?"}],
    )
    raise ValueError("Something went wrong")


@baserun.trace
def openai_completion(prompt="Human: say this is a test\nAssistant: ") -> str:
    client = OpenAI()
    completion = client.completions.create(model="text-davinci-003", prompt=prompt)
    content = completion.choices[0].text
    baserun.check_includes("openai_chat_async_streaming.content", content, "test")
    return content


@baserun.trace
async def openai_completion_async(
    prompt="Human: say this is a test\nAssistant: ",
) -> str:
    client = AsyncOpenAI()
    completion = await client.completions.create(model="text-davinci-003", prompt=prompt)
    content = completion.choices[0].text
    baserun.check_includes("openai_chat_async_streaming.content", content, "test")
    return content


@baserun.trace
def openai_completion_streaming(prompt="Human: say this is a test\nAssistant: ") -> str:
    client = OpenAI()
    completion_generator = client.completions.create(model="text-davinci-003", prompt=prompt, stream=True)

    content = ""
    for chunk in completion_generator:
        if new_content := chunk.choices[0].text:
            content += new_content

    baserun.check_includes("openai_chat_async_streaming.content", content, "test")
    return content


@baserun.trace
async def openai_completion_async_streaming(
    prompt="Human: say this is a test\nAssistant: ",
) -> str:
    client = AsyncOpenAI()
    completion_generator = await client.completions.create(model="text-davinci-003", prompt=prompt, stream=True)
    content = ""
    async for chunk in completion_generator:
        if new_content := chunk.choices[0].text:
            content += new_content

    baserun.check_includes("openai_chat_async_streaming.content", content, "test")
    return content


@baserun.trace
def openai_threaded():
    threads = [
        Thread(
            target=baserun.thread_wrapper(openai_chat_unwrapped),
            args=("What is the capital of the state of Georgia?",),
        ),
        Thread(
            target=baserun.thread_wrapper(openai_chat_unwrapped),
            args=("What is the capital of the California?",),
            kwargs={"top_p": 0.5},
        ),
        Thread(
            target=baserun.thread_wrapper(openai_chat_unwrapped),
            args=("What is the capital of the Montana?",),
            kwargs={"temperature": 1},
        ),
    ]
    [t.start() for t in threads]
    [t.join() for t in threads]


@baserun.trace
def openai_chat_response_format(prompt="What is the capital of the US?") -> str:
    client = OpenAI()
    completion = client.chat.completions.create(
        model="gpt-4-1106-preview",
        messages=[
            {"role": "system", "content": "Respond to the following question in JSON"},
            {"role": "user", "content": prompt},
        ],
        response_format={"type": "json_object"},
    )
    content = completion.choices[0].message.content
    baserun.check_includes("openai_chat.content", content, "Washington")
    return content


@baserun.trace
def openai_chat_seed(prompt="What is the capital of the US?") -> str:
    client = OpenAI()
    completion = client.chat.completions.create(
        model="gpt-4-1106-preview",
        messages=[{"role": "user", "content": prompt}],
        seed=1234,
    )
    content = completion.choices[0].message.content
    baserun.check_includes("openai_chat.content", content, "Washington")
    return content


def openai_contextmanager(prompt="What is the capital of the US?", name: str = "This is a run that is named") -> str:
    client = OpenAI()
    with baserun.start_trace(name=name) as run:
        completion = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
        )
        content = completion.choices[0].message.content
        run.result = content


TEMPLATES = {"Question & Answer": "Answer the following question in the form of a limerick: {question}"}


@baserun.trace
def use_template(question="What is the capital of the US?", template_name: str = "Question & Answer"):
    prompt = baserun.format_prompt(
        template_string=TEMPLATES.get(template_name),
        template_name=template_name,
        parameters={"question": question},
    )

    client = OpenAI()
    completion = client.chat.completions.create(
        model="gpt-4-1106-preview",
        messages=[{"role": "system", "content": prompt}],
        seed=1234,
    )
    content = completion.choices[0].message.content
    baserun.check_includes("openai_chat.content", content, "Washington")
    return content


@baserun.trace
async def use_template_async(question="What is the capital of the US?", template_name: str = "Question & Answer"):
    prompt = await baserun.aformat_prompt(
        template_string=TEMPLATES.get(template_name),
        template_name=template_name,
        parameters={"question": question},
    )

    client = AsyncOpenAI()
    completion = await client.chat.completions.create(
        model="gpt-4-1106-preview",
        messages=[{"role": "system", "content": prompt}],
        seed=1234,
    )
    content = completion.choices[0].message.content
    baserun.check_includes("openai_chat.content", content, "Washington")
    return content


@baserun.trace
def display_templates():
    templates = baserun.get_templates()
    for template_name, template in templates.items():
        print(template_name)
        padded_string = "\n  | ".join(template.active_version.template_string.split("\n"))
        print(f"| Tag: {template.active_version.tag}")
        print(f"| Template: ")
        print(padded_string.strip())
        print("")

    return "Done"


@baserun.trace
def use_sessions(prompt="What is the capital of the US?", user_identifier="example@test.com") -> str:
    client = OpenAI()
    with baserun.with_session(user_identifier=user_identifier):
        completion = client.chat.completions.create(
            model="gpt-4-1106-preview",
            messages=[{"role": "user", "content": prompt}],
        )
        content = completion.choices[0].message.content
        baserun.check_includes("openai_chat.content", content, "Washington")
        baserun.log(f"OpenAI Chat Results", payload={"result": content, "input": prompt})
        return content


@baserun.trace
async def use_annotation(question="What is the capital of the US?") -> str:
    client = OpenAI()

    completion = client.chat.completions.create(
        model="gpt-4-1106-preview",
        messages=[{"role": "user", "content": question}],
    )
    content = completion.choices[0].message.content

    annotation = baserun.annotate(completion.id)
    annotation.feedback(
        name="use_annotation_feedback",
        score=0.8,
        metadata={"comment": "This is correct but not concise enough"},
    )
    annotation.check_includes("openai_chat.content", "Washington", content)
    annotation.log(f"OpenAI Chat Results", metadata={"result": content, "input": question})
    await annotation.asubmit()

    return content


def use_langchain(question="What is the capital of the US?") -> str:
    from baserun.instrumentation.langchain import BaserunCallbackHandler
    from langchain.chat_models import ChatOpenAI
    from langchain_core.messages import HumanMessage

    chat = ChatOpenAI(callbacks=[BaserunCallbackHandler()])
    messages = [HumanMessage(content=question)]
    response = chat.invoke(messages)
    return response.content


def use_langchain_tools(question="What is the capital of the US?") -> str:
    # Note: To run this, you must `pip install wikipedia langchain`
    from langchain.chat_models import ChatOpenAI
    from langchain_core.messages import HumanMessage
    from langchain.tools.render import format_tool_to_openai_tool
    from langchain.agents import load_tools
    from baserun.instrumentation.langchain import BaserunCallbackHandler

    chat = ChatOpenAI(callbacks=[BaserunCallbackHandler()])
    tools = [format_tool_to_openai_tool(t) for t in load_tools(["wikipedia"], llm=chat)]
    messages = [HumanMessage(content=question)]
    response = chat.invoke(messages, tools=tools)
    return json.dumps(response.additional_kwargs.get("tool_calls"))


def use_langchain_chain(question="What is the capital of {location}?") -> str:
    from langchain.chat_models import ChatOpenAI
    from langchain_core.prompts import PromptTemplate
    from langchain.chains import LLMChain
    from baserun.instrumentation.langchain import BaserunCallbackHandler

    chat = ChatOpenAI()

    chain = LLMChain(
        llm=chat,
        prompt=PromptTemplate(template=question, input_variables=["location"]),
        callbacks=[BaserunCallbackHandler()],
    )
    chain.run(location="the US")
    chain.run(location="California")
    response = chain.run(location="Georgia")
    return response


def use_langchain_agent_tools(question="Using Wikipedia, look up the population of {location} as of 2023.") -> str:
    # Note: To run this, you must `pip install wikipedia langchain`
    from langchain.chat_models import ChatOpenAI
    from langchain_core.prompts import PromptTemplate
    from langchain.agents import initialize_agent
    from langchain.agents import AgentType
    from langchain.agents import load_tools
    from langchain_core.callbacks import BaseCallbackManager
    from baserun.instrumentation.langchain import BaserunCallbackHandler

    chat = ChatOpenAI()
    tools = load_tools(["wikipedia"])
    prompt_template = PromptTemplate(template=question, input_variables=["location"])

    agent = initialize_agent(
        tools,
        chat,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        callback_manager=BaseCallbackManager(handlers=[BaserunCallbackHandler()]),
    )
    response = agent.run(prompt_template.format_prompt(location="California"))
    return response


def call_function(functions, function_name: str, parsed_args: argparse.Namespace):
    function_to_call = functions.get(function_name)
    if function_to_call is None:
        function_to_call = {f: globals().get(f) for f in globals()}.get(function_name)

    if inspect.iscoroutinefunction(function_to_call):
        if parsed_args.prompt:
            result = asyncio.run(function_to_call(parsed_args.prompt))
        else:
            result = asyncio.run(function_to_call())
    else:
        if parsed_args.prompt:
            result = function_to_call(parsed_args.prompt)
        else:
            result = function_to_call()

    print(result)
    return result


# Allows you to call any of these functions, e.g. python tests/testing_functions.py openai_chat_functions_streaming
if __name__ == "__main__":
    from dotenv import load_dotenv
    from baserun import Baserun

    load_dotenv()
    openai.api_key = os.environ.get("OPENAI_API_KEY")
    Baserun.init()

    parser = argparse.ArgumentParser(description="Execute a function with a prompt.")
    parser.add_argument("function_to_call", type=str, help="Name of the function to call")
    parser.add_argument("--prompt", type=str, help="Prompt to pass to the function", default=None)

    parsed_args = parser.parse_args()

    # Resolve the string function name to the function object
    function_name = parsed_args.function_to_call
    global_variables = {f: globals().get(f) for f in globals()}
    traced_functions = {n: f for n, f in global_variables.items() if callable(f) and f.__name__ == "wrapper"}
    if function_name == "all":
        for name, func in traced_functions.items():
            print(f"===== Calling function {name} =====\n")
            try:
                result = call_function(traced_functions, name, parsed_args)
                print(f"----- {name} result:\n{result}\n-----")
            except Exception as e:
                traceback.print_exception(e)
    else:
        call_function(traced_functions, function_name, parsed_args)
