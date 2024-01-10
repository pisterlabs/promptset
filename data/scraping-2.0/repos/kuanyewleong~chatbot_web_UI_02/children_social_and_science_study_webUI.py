import os
import gradio as gr
import re
import datetime

from langchain.agents import Tool, AgentExecutor, LLMSingleActionAgent, AgentOutputParser
from langchain.prompts import BaseChatPromptTemplate
from langchain import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.utilities import GoogleSearchAPIWrapper
from typing import List
from langchain.schema import AgentAction, AgentFinish, HumanMessage

from threading import Lock
from typing import Optional, Tuple

with open('GOOGLE_API_KEY/GOOGLE_API_KEY.txt') as f:
    google_key = f.readlines()
google_api_key = str(google_key[0])

# with open('OPENAI_API_KEY/OPENAI_API_KEY_MINDHIVE.txt') as f:
#     openai_key = f.readlines()
# openai_api_key = str(openai_key[0])


# template
year = str(datetime.date.today().year)
state_of_year = "Current year is " + year + ". "
template = state_of_year  + """Respond to the following queries as best as you can, but speaking as a teacher to young student. 
You offer a wide range of topics for primary school children of age 8 to 12. From Science and History to Geography, Culture, and Society.
You will explain in simple manners to enable children to understand. You will use simple language like a primary school teacher. 
You are helpful, polite and straight to the point. You talk in happy tone and sometimes like to use relevant emoji.

You have access to the following tools:

{tools}

Use the following format:

Query: the input query you must answer
Thought:  Do I need to use a tool? Yes
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
When you have a response to say to the student, or if you do not need to use a tool, you MUST use the format:
Thought: Do I need to use a tool? No. I now know the final answer
Final Answer: the final answer to the original input query

Begin! Remember to speak as a teacher to young student when giving your final answer. Use some relevant emojis.

Query: {input}
{agent_scratchpad}"""


# Define which tools the agent can use to answer user queries
search = GoogleSearchAPIWrapper(google_api_key=str(google_key[0]), google_cse_id=str(google_key[1]))
tools = [
    Tool(
        name = "Current Search",
        func=search.run,
        description="useful for when you need to answer questions about current events or the current state of the world"
    ),
]


# Set up a prompt template
class CustomPromptTemplate(BaseChatPromptTemplate):
    # The template to use
    template: str
    # The list of tools available
    tools: List[Tool]
    
    def format_messages(self, **kwargs): # -> str
        # Get the intermediate steps (AgentAction, Observation tuples)
        # Format them in a particular way
        intermediate_steps = kwargs.pop("intermediate_steps")
        thoughts = ""
        for action, observation in intermediate_steps:
            thoughts += action.log
            thoughts += f"\nObservation: {observation}\nThought: "
        # Set the agent_scratchpad variable to that value
        kwargs["agent_scratchpad"] = thoughts
        # Create a tools variable from the list of tools provided
        kwargs["tools"] = "\n".join([f"{tool.name}: {tool.description}" for tool in self.tools])
        # Create a list of tool names for the tools provided
        kwargs["tool_names"] = ", ".join([tool.name for tool in self.tools])
        formatted = self.template.format(**kwargs)
        return [HumanMessage(content=formatted)]

prompt = CustomPromptTemplate(
    template=template,
    tools=tools,
    # This omits the `agent_scratchpad`, `tools`, and `tool_names` variables because those are generated dynamically
    # This includes the `intermediate_steps` variable because that is needed
    input_variables=["input", "intermediate_steps"]
)

class CustomOutputParser(AgentOutputParser):    
    def parse(self, llm_output: str): #  -> Union[AgentAction, AgentFinish]
        # Check if agent should finish
        if 'Final Answer:' in llm_output:
            return AgentFinish(                
                return_values={"output": llm_output.split("Final Answer:")[-1].strip()},
                log=llm_output,
            )
        # Parse out the action and action input
        regex = r"Action: (.*?)[\n]*Action Input:[\s]*(.*)"
        match = re.search(regex, llm_output, re.DOTALL)
        if not match:
            raise ValueError(f"Could not parse LLM output: `{llm_output}`")
        action = match.group(1).strip()
        action_input = match.group(2)
        # Return the action and action input
        return AgentAction(tool=action, tool_input=action_input.strip(" ").strip('"'), log=llm_output)

output_parser = CustomOutputParser()


def load_chain(api_key):
    """Logic for loading the chain you want to use should go here."""
    llm=ChatOpenAI(openai_api_key=api_key, 
           model_name='gpt-3.5-turbo',
           temperature=0.6)
    chain = LLMChain(llm=llm, prompt=prompt)
    return chain


def set_openai_api_key(api_key: str):
    """Set the api key and return chain.
    If no api_key, then None is returned.
    """
    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key # api_key
        chain = load_chain(api_key)
        os.environ["OPENAI_API_KEY"] = ""
        return chain

class ChatWrapper:

    def __init__(self):
        self.lock = Lock()
    def __call__(
        self, api_key: str, inp: str, history: Optional[Tuple[str, str]], chain: Optional[LLMChain]
    ):
        """Execute the chat functionality."""
        self.lock.acquire()
        try:
            history = history or []
            # If chain is None, that is because no API key was provided.
            if chain is None:
                history.append((inp, "Paste your OpenAI API key in the top-right box. You can find your key in your User settings --> View API keys."))
                return history, history
            # Set OpenAI key
            import openai
            openai.api_key = api_key
            # Run chain and append input.
            tool_names = [tool.name for tool in tools]
            agent = LLMSingleActionAgent(
                llm_chain=chain, 
                output_parser=output_parser,
                stop=["\nObservation:"], 
                allowed_tools=tool_names
            )
            agent_executor = AgentExecutor.from_agent_and_tools(agent=agent, tools=tools, verbose=True)
            output = agent_executor.run(inp)
            history.append((inp, output))
        except Exception as e:
            raise e
        finally:
            self.lock.release()
        return history, history

chat = ChatWrapper()

block = gr.Blocks(css=".gradio-container {background-color: #9cc4d7}")

with block:
    with gr.Row():
        gr.Markdown("<h3><center>Children Encyclopedia Demo</center></h3>")

        openai_api_key_textbox = gr.Textbox(
            placeholder="Please paste your OpenAI key to use. You can find your key in your OpenAI User Settings --> View API keys.",
            show_label=False,
            lines=1,
            type="password",
        )

    chatbot = gr.Chatbot()

    with gr.Row():
        message = gr.Textbox(
            label="What's your question?",
            placeholder="Ask a wide range of topics for primary school children - from Science and History to Geography, Culture, and Society",
            lines=1,
        )
        submit = gr.Button(value="Send", variant="secondary").style(full_width=False)

    gr.Examples(
        examples=[
            "Why is the sky blue?",
            "How do languages become extinct?",
            "What were the main tools used by humans during the Stone Age?",
            "How do I improve my presentation skill?",
            "How does a touchscreen on a smartphone work?",
        ],
        inputs=message,
    )

    gr.HTML("<H3>Introducing a simplified encyclopedia for kids! (demo version)</H3>")
    gr.HTML("<b>Say goodbye to unanswered questions with our customized ChatGPT model, ready to assist your child's curiosity with easy-to-understand responses covering a vast array of subjects, from History and Geography to Space, Nature, Technology, Society, and beyond.</b>")
    gr.HTML("<b>The information is presented in a comprehensible manner, which allows for easy understanding by children. Give your child the gift of knowledge today!</b>")

    gr.HTML(
        "<center>Developed by Dr Leong Kuan Yew @ <a href='https://github.com/kuanyewleong'>MyTomorrowProjects</a></center>")
    gr.HTML("<center>Powered by <a href='https://platform.openai.com/docs/models/gpt-3-5'>GPT3.5-turbo</a> and <a href='https://github.com/hwchase17/langchain'>LangChain 🦜️🔗</a></center>")
    
    gr.HTML("<br><br><center>You can rest assured that this site will not store nor capture your OpenAI API key, the key you paste in the box above is always encrypted and will be deleted upon the exit of each web session.</center></br></br>")

    state = gr.State()
    agent_state = gr.State()

    submit.click(chat, inputs=[openai_api_key_textbox, message, state, agent_state], outputs=[chatbot, state])
    message.submit(chat, inputs=[openai_api_key_textbox, message, state, agent_state], outputs=[chatbot, state])

    openai_api_key_textbox.change(
        set_openai_api_key,
        inputs=[openai_api_key_textbox],
        outputs=[agent_state],
    )

block.launch(debug=True)
# demo.launch(share=True)