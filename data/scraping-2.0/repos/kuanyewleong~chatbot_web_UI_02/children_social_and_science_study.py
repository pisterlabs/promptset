from langchain.agents import Tool, AgentExecutor, LLMSingleActionAgent, AgentOutputParser
from langchain.prompts import BaseChatPromptTemplate
from langchain import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.utilities import GoogleSearchAPIWrapper
from typing import List
from langchain.schema import AgentAction, AgentFinish, HumanMessage

import re
import openai

with open('GOOGLE_API_KEY/GOOGLE_API_KEY.txt') as f:
    google_key = f.readlines()
google_api_key = str(google_key[0])

with open('OPENAI_API_KEY/OPENAI_API_KEY_MINDHIVE.txt') as f:
    openai_key = f.readlines()
openai_api_key = str(openai_key[0])
openai.api_key = openai_api_key

# with open('template_children_study.txt') as f:
#     template = f.readlines()
# template = str(template)

template = """Respond to the following queries as best as you can, but speaking as a teacher to young student. 
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

def remove_dialogue(text):
    # Define regular expressions for dialogue structures
    pattern = re.compile(r"AI:|Assistant:")    
    text_return = re.sub(pattern, "", text)       
    return text_return

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

llm=ChatOpenAI(openai_api_key=openai_api_key, 
           model_name='gpt-3.5-turbo',
           temperature=0.7)

# LLM chain consisting of the LLM and a prompt
llm_chain = LLMChain(llm=llm, prompt=prompt)
tool_names = [tool.name for tool in tools]
agent = LLMSingleActionAgent(
    llm_chain=llm_chain, 
    output_parser=output_parser,
    stop=["\nObservation:"], 
    allowed_tools=tool_names
)

agent_executor = AgentExecutor.from_agent_and_tools(agent=agent, tools=tools, verbose=True)

def opening_text():    
    topic = "Say the following line in some creative way:\n"
    open = "Hi there, how may I help you?\n"    
    all_prompt = topic + open
    
    # Generate a text completion
    response = openai.Completion.create(
    model="text-ada-001",
    prompt = all_prompt,    
    temperature=0.9,
    max_tokens=32,
    top_p=1,
    frequency_penalty=0,
    presence_penalty=0.
    )
    # Print the generated text completion
    print(response.choices[0].text)


opening_text()
while (True):
    user_input = input()
    output = agent_executor.run(user_input)
    print(output)
    print("\nDo you have more questions?\n")

    
# while (True):    
#     human_input = input()    
#     generated_text = chatgpt_chain.predict(human_input=human_input)
#     print(remove_dialogue(generated_text))
    
# agent_chain = initialize_agent(tools, llm, agent="conversational-react-description", verbose=False, memory=memory)

# while (True):
#     user_input = input()
#     output = agent_chain.run(user_input)
#     print(output)

    
# output = chatgpt_chain.predict(human_input="Tell me a joke.")
# print(output)



# def openai_chat(customer_input):    
#     # Greet and get prompt from customer
#     system_start = "You are an AI assistant. The assistant is helpful, creative, polite, and very friendly."
#     assistant_start = "Hi there, may I help you?"        
#     # have to remove " from the text to avoid issue with json object structure
#     # replace with ' instead
#     translate_table = str.maketrans('"', "'")
#     customer_input = customer_input.translate(translate_table)
        
#     system_head = '{"role": "system", "content": '
#     asst_head =  '{"role": "assistant", "content": '
#     customer_head =  '{"role": "user", "content": '

#     all_prompt = [
#             json_obj_maker(system_head, system_start),
#             json_obj_maker(asst_head, assistant_start),
#             json_obj_maker(customer_head, customer_input)       
#         ]
    
#     response = openai.ChatCompletion.create(
#     model="gpt-3.5-turbo",
#     temperature=0.9,
#     max_tokens=128,
#     messages=all_prompt,
#     top_p=1,
#     frequency_penalty=0,
#     presence_penalty=0,
#     stop=["That's all", "Thank you", "bye"]
#     )
#     message = response['choices'][0]['message']['content']
#     return message.strip()

# def get_bot(input, history=[]):    
#     history += [input] # context += [query]
#     # print("History+=input: ", history)
#     prompt = '\n\n'.join(history)[:4096]
#     # print("prompt: ", prompt)
#     output = openai_chat(prompt)
#     # print("output: ", output)
#     history += [output]
#     # print("history+=[output]: ", history)
#     responses = [(u,b) for u,b in zip(history[::2], history[1::2])]    

#     return responses, history

# with gr.Blocks(css=".gradio-container {background-color: powderblue}" "#chatbot .overflow-y-auto{height:500px}") as demo:
#     with open('welcome_text.txt') as f:
#         welcome_note = f.readlines()
#     chatbot = gr.Chatbot([(" ==== Social and Science Studies for Children ==== \n", None), 
#                           (str(welcome_note[0]), None)], elem_id="chatbot")
#     state = gr.State([])

#     with gr.Row(variant='compact'):
#         with gr.Column(scale=0.85):
#             txt = gr.Textbox(show_label=False, placeholder="Enter text and press enter ... ").style(container=False)
        
#     txt.submit(get_bot, [txt, state], [chatbot, state])
#     txt.submit(lambda :"", None, txt)        
    
# demo.launch()
# demo.launch(debug=True)
