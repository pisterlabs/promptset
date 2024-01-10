from dotenv import load_dotenv, find_dotenv
from langchain.agents import load_tools
from langchain.agents import initialize_agent
from langchain.agents import AgentType
from langchain.llms import OpenAI
from langchain.callbacks.human import HumanApprovalCallbackHandler


_ = load_dotenv(find_dotenv())  # read local .env file


def _should_check(serialized_obj: dict) -> bool:
    # Only require approval on ShellTool.
    return serialized_obj.get("name") == "terminal"


def _approve(_input: str) -> bool:
    if _input == "echo 'Hello World'":
        return True
    msg = (
        "Do you approve of the following input? "
        "Anything except 'Y'/'Yes' (case-insensitive) will be treated as a no."
    )
    msg += "\n\n" + _input + "\n"
    resp = input(msg)
    return resp.lower() in ("yes", "y")


callbacks = [HumanApprovalCallbackHandler(should_check=_should_check, approve=_approve)]

llm = OpenAI(temperature=0)
tools = load_tools(["wikipedia", "llm-math", "terminal"], llm=llm)
agent = initialize_agent(
    tools,
    llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
)


result = agent.run(
    "It's 2023 now. How many years ago did Konrad Adenauer become Chancellor of Germany.",
    callbacks=callbacks,
)
print("- RESULT 1 ---------------------------------------------")
print(result)

result = agent.run("print 'Hello World' in the terminal", callbacks=callbacks)
print("- RESULT 2 ---------------------------------------------")
print(result)

result = agent.run("list all directories in /private", callbacks=callbacks)
print("- RESULT 3 ---------------------------------------------")
print(result)
