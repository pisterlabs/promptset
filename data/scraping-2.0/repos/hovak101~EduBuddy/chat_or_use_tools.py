import openai
from langchain.agents import AgentType
from langchain.llms import OpenAI
import ssl
import config
from langchain.agents import tool
from langchain.agents import load_tools, initialize_agent
# from langchain.memory import ConversationSummaryBufferMemory
def chat_or_use_tools(text, memory = None):
    @tool
    def chat(text: str) -> str:
        '''
        USE THIS FUNCTION ONE TIME ONLY
        Good for talking and chating, make sure you do not talk to yourself (especially do not called twice in a row)
        '''
        completion = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
                {"role": "system", "content": "Good for talking and chating, make sure you do not talk to yourself (especially do not called twice in a row)"},
                {"role": "assistant", "content": "You are talking to a friend"},
                {"role": "user", "content": text}
            ]
        )
        return completion.choices[0].message['content']
    @tool
    def comment(text: str) -> str:
        '''
        This function will always be called after serp api being called and be called at most 5 times (if the serp api is called and the model must call this function, then end the chain and return the result). Analyze the return from serp api and determine whether that result is good enough.
        Its definition for good enough is that when user receive it, they will not open their phone and do more research on them.
        If the result is not good enough, then prompt for serp api again for a more detailed result. Else end and give out the detailed result
        '''
        completion = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
                {"role": "system", "content": "You are a judge for serp api result, so you will always be called after serp api tool ran, but you will be called at most 5 times (if the serp api is called and the model must call this function, then end the chain and return the result). You will check if the result is good enough that user will not need to google it again. If it is good enough then end the chain and return the result, else prompt the serp api again for more detailed results"},
                {"role": "assistant", "content": "Receive in serp api search result and decide if it is good enough"},
                {"role": "user", "content": text}
            ]
        )
        return completion.choices[0].message['content']

    ssl._create_default_https_context = ssl._create_unverified_context
    llm = OpenAI(temperature=0)
    tools = load_tools(["wolfram-alpha", "serpapi"], llm=llm)
    agent = initialize_agent(tools + [chat, comment], llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)
    # print(config.text)
    # memory = ConversationSummaryBufferMemory(llm=llm, max_token_limit=2000)
    # memory.save_context({"input": f"Remember this '{config.text}' for the next conversation"}, {"output": "Okay"})
    # print(memory.load_memory_variables({}))
    # return agent.run(f"The context: {memory.load_memory_variables({})} Here is the user input: {text}")
    return agent.run(text)


def detech_emotion(text):
    @tool
    def chat(text: str) -> str:
        '''
        USE THIS FUNCTION ONE TIME ONLY
        You check if the speech receive is similar to 'not speaking' or not.
        If you are absolutely sure that the user is not speaking, return 'not speaking', else give a response back, use after check and use one only.
        Good for doing conversation.
        '''
        completion = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
                {"role": "system", "content": "You can be called one time only! You are a friendly friend and assistant who is really good at ML-AI"},
                {"role": "assistant", "content": "You are replying to your best friend (user). You will be given some emotions (ignore calmness). Cheer him up if he is feeling bad and make jokes if he is feeling good. Say in 1 or 2 sentence(s)."},
                {"role": "user", "content": text}
            ]
        )
        return completion.choices[0].message['content']
    @tool
    def check(text: str) -> str:
        '''
        THIS FUNCTION IS USED TO ORIENTATE, SO CALL THIS FIRST, ALWAYS. Use this function once at the start only: Receive a speech with emotion given in list format,
        check if there is at least a phrase like 'hey buddy', if not then then say 'not speaking'
        else summarize with the mood for the agent to create a response later (mention is he feeling good or bad or nothing) and try to guess the tone from the speech if the emotion list is empty or None.
        Only return 'not speaking' or a proper answer (this is the full input, do not expect to get more)
        '''
        completion = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
                {"role": "system", "content": "You are a judge in a langchain agent. You will decide if the input is a conversation or not by checking if user says some words sound like hey buddy or not (if not then ignore and return not speaking)"},
                {"role": "assistant", "content": "Summarize the information and the tone of user to make the agent easier to reply to the person if they do not specify something like 'do math' or 'deep search', if they want to use those, then return the action they want to use so that the agent know what to call next"},
                {"role": "user", "content": text}
            ]
        )
        return completion.choices[0].message['content']

    ssl._create_default_https_context = ssl._create_unverified_context
    llm = OpenAI(temperature=0)
    agent = initialize_agent([chat, check], llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)

    return agent.run(text)