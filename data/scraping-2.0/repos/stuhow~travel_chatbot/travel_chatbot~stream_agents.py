
from travel_chatbot.chains import check_conversation_stage, bq_check_conversation_stage
from travel_chatbot.prompts import customize_prompt
from langchain.agents import OpenAIFunctionsAgent, AgentExecutor
from langchain.chat_models import ChatOpenAI
from travel_chatbot.tools import get_bq_tools
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

def run_francis(input,
                conversation_history,
                user_travel_details,
                list_of_interests,
                interest_asked,
                tools,
                asked_for):

    llm = ChatOpenAI(temperature=0.9, model="gpt-3.5-turbo")

    # user_input = f"User: {input}"

    # conversation_history.append(user_input)

    conversation_stage, user_travel_details = check_conversation_stage(conversation_history,
                                                                       user_travel_details,
                                                                       list_of_interests,
                                                                       interest_asked,
                                                                       asked_for)

    final_prompt = customize_prompt(conversation_history, conversation_stage)


    # Create the agent
    agent = OpenAIFunctionsAgent(llm=llm,
                                 tools=tools,
                                 prompt=final_prompt)
    # agent = OpenAIMultiFunctionsAgent(llm=llm, tools=tools, prompt=final_prompt)

    # Run the agent with the actions
    agent_executor = AgentExecutor(agent=agent,
                                   tools=tools,
                                   verbose=False,
                                   max_iterations=5)

    francis = agent_executor.run(input)
    # francis1 = f"Francis: {francis}"
    # conversation_history.append(francis1)

    return francis, user_travel_details


def bq_run_francis(input,
                conversation_history,
                user_travel_details,
                list_of_interests,
                interest_asked,
                tools,
                asked_for,
                solution_presented):

    llm = ChatOpenAI(temperature=0.9, model="gpt-3.5-turbo")

    conversation_stage, new_user_travel_details, new_list_of_interests, found_itineraries = bq_check_conversation_stage(conversation_history,
                                                                       user_travel_details,
                                                                       list_of_interests,
                                                                       interest_asked,
                                                                       asked_for,
                                                                       solution_presented)

    final_prompt = customize_prompt(conversation_history, conversation_stage)
    print(final_prompt)
    print("gathering tools")
    tools = get_bq_tools(found_itineraries, new_list_of_interests, new_user_travel_details)

    print("running agent")
    # Create the agent
    agent = OpenAIFunctionsAgent(llm=llm,
                                 tools=tools,
                                 prompt=final_prompt)
    # agent = OpenAIMultiFunctionsAgent(llm=llm, tools=tools, prompt=final_prompt)

    # Run the agent with the actions
    agent_executor = AgentExecutor(agent=agent,
                                   tools=tools,
                                   verbose=False,
                                   max_iterations=5,
                                   early_stopping_method="generate")

    francis = agent_executor.run(input)

    return francis, new_user_travel_details


def bq_stream_francis(input,
                conversation_history,
                user_travel_details,
                list_of_interests,
                interest_asked,
                tools,
                asked_for,
                solution_presented):

    llm = ChatOpenAI(temperature=0.9,
                     model="gpt-3.5-turbo",
                     streaming=True,
                     callbacks=[StreamingStdOutCallbackHandler()]
                     )

    conversation_stage, new_user_travel_details, new_list_of_interests, found_itineraries = bq_check_conversation_stage(conversation_history,
                                                                       user_travel_details,
                                                                       list_of_interests,
                                                                       interest_asked,
                                                                       asked_for,
                                                                       solution_presented)

    final_prompt = customize_prompt(conversation_history, conversation_stage)
    print(final_prompt)
    print("gathering tools")
    tools = get_bq_tools(found_itineraries, new_list_of_interests, new_user_travel_details)

    print("running agent")
    # Create the agent
    agent = OpenAIFunctionsAgent(llm=llm,
                                 tools=tools,
                                 prompt=final_prompt)
    # agent = OpenAIMultiFunctionsAgent(llm=llm, tools=tools, prompt=final_prompt)

    # Run the agent with the actions
    agent_executor = AgentExecutor(agent=agent,
                                   tools=tools,
                                   verbose=False,
                                   return_intermediate_steps=False,
                                   max_iterations=5,
                                   early_stopping_method="generate")

    francis = agent_executor.run(input)

    return agent_executor, new_user_travel_details
