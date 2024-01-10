from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage, AIMessage, SystemMessage
from dotenv import load_dotenv

load_dotenv()

chat_llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.7)

#single & multiple prompts
single_prompt = "the first person lands on the moon is"
multiple_prompts = [SystemMessage(content="you are a sarcastic historian"),
                    HumanMessage(content="the first person lands on the moon is")]
# single_response = chat_llm.predict(single_prompt)
ai_response = chat_llm.predict_messages(multiple_prompts)
print(ai_response)

#Using "generate" method, instead of "predict_messages"
# batch_messages = [
#             [SystemMessage(content="you are a standup comedian"),
#             HumanMessage(content="tell me a joke about a stupid person")],
#             [SystemMessage(content="you are a pizza delivery guy"),
#             HumanMessage(content="tell me a joke about your job")],
#             [SystemMessage(content="you are a sarcastic historian"),
#             HumanMessage(content="tell me a joke about Crimea annexation")],
#         ]
# response = chat_llm.generate(batch_messages)
# print(response.generations[0][0].text)
# print(response.generations[1][0].text)
# print(response.generations[2][0].text)



