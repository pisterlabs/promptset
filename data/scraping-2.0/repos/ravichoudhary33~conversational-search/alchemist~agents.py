# from langchain.chat_models import ChatOpenAI
# from langchain.agents import initialize_agent
#
# from tools import RetrievalTool
# from memory import conversational_memory
#
# llm = ChatOpenAI(
#         temperature=0,
#         model_name='gpt-3.5-turbo'
# )
#
# tools = [RetrievalTool()]
#
# def create_agent(llm, tools):
#     agent = initialize_agent(
#         agent='chat-conversational-react-description',
#         tools=tools,
#         llm=llm,
#         verbose=True,
#         max_iterations=3,
#         early_stopping_method='generate',
#         memory=conversational_memory,
#         return_intermediate_steps=True
#     )
#
#     return agent