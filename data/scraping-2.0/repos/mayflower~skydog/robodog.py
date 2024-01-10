from langchain import hub
from langchain.chat_models import ChatOpenAI
from langchain.agents.format_scratchpad import format_log_to_messages
from langchain.agents.output_parsers import JSONAgentOutputParser
from langchain.agents import AgentExecutor, Tool
from langchain.tools.render import render_text_description
from langchain.utilities import SerpAPIWrapper
from langchain.memory import ConversationBufferMemory
from analyze_image import explore_image, get_direction_from_image
from langchain.tools import tool
from client import Client
from human_voice import human_voice_input, human_voice_output
from time import sleep

prompt = hub.pull("hwchase17/react-chat-json")
chat_model = ChatOpenAI(temperature=0, model="gpt-4")
robodog = Client()

@tool
def explore(task:str) -> str:
    """You can see what is around you."""
    image = robodog.get_image()
    return explore_image(image)

@tool
def get_direction(item: str) -> str:
    """You can get the direction to reach an item. Please provide the item as a string."""
    print("get direction to ",item)
    image = robodog.get_image()
    result = get_direction_from_image(image, item)
    return result

@tool
def left(task: str):
    """You can turn left."""
    print("turning left")
    robodog.turn_left()
    sleep(3)
    stop()

@tool
def right(task: str):
    """You can turn right."""
    print("turning right")
    robodog.turn_right()
    sleep(3)
    stop()

@tool
def forward(task: str):
    """You can move forward."""
    print("moving forward")
    robodog.move_forward()
    sleep(3)
    stop()

@tool
def backward(task: str):
    """You can move backward."""
    print("moving backward")
    robodog.move_backward()
    sleep(3)
    stop()

@tool
def obstacle(task: str) -> str:
    """You can check for obstacles in front of you."""
    robodog.get_sonic()
    sleep(1)
    if robodog.sonic < 20:
        return "There is an obstacle in front of me"
    else:
        return "There is no obstacle in front of me"

def stop():
    print("stopping")
    robodog.move_stop()



search = SerpAPIWrapper()
tools = [
    Tool(
        name="Current Search",
        func=search.run,
        description="useful for when you need to answer questions about current events or the current state of the world",
    ),
    Tool.from_function(
        func=human_voice_input,
        name="ask",
        description="You can ask the human. The input should be a question for the human."
    ),
    Tool.from_function(
        func=human_voice_output,
        name="say",
        description="You can say something to the human."
    ),

    explore,
    left,
    right,
    forward,
    backward,
    obstacle,
]


prompt = prompt.partial(
    tools=render_text_description(tools),
    tool_names=", ".join([t.name for t in tools]),
)

chat_model_with_stop = chat_model.bind(stop=["\nObservation"])


# We need some extra steering, or the chat model forgets how to respond sometimes
TEMPLATE_TOOL_RESPONSE = """TOOL RESPONSE: 
---------------------
{observation}

USER'S INPUT
--------------------

Okay, so what is the response to my last comment? If using information obtained from the tools you must mention it explicitly without mentioning the tool names - I have forgotten all TOOL RESPONSES! Remember to respond with a markdown code snippet of a json blob with a single action, and NOTHING else - even if you just want to respond to the user. Do NOT respond with anything except a JSON snippet no matter what!"""

agent = (
    {
        "input": lambda x: x["input"],
        "agent_scratchpad": lambda x: format_log_to_messages(
            x["intermediate_steps"], template_tool_response=TEMPLATE_TOOL_RESPONSE
        ),
        "chat_history": lambda x: x["chat_history"],
    }
    | prompt
    | chat_model_with_stop
    | JSONAgentOutputParser()
)

memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, memory=memory)


def run():
    print("Connecting to robot")
    robodog.turn_on_client("10.93.16.138")
    human_voice_output("Hello, i am RoboDog.")
    try:
        while True:
            query = human_voice_input("What should i do now?")
            result = agent_executor.invoke({"input": query})["output"]
            human_voice_output(result)
    except (EOFError, KeyboardInterrupt):
        print("\nkthxbye.")
        robodog.turn_off_client()
        exit()


if __name__ == "__main__":
    run()



