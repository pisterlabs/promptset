# from langchain import ConversationChain, PromptTemplate, LLMChain
# from langchain.memory import ConversationBufferWindowMemory
# from langchain.chat_models import ChatOpenAI
#
# # Customize the LLM template
# template = """Assistant is a large language model trained by OpenAI.
#
# Human: {human_input}
# Assistant:"""
#
# prompt = PromptTemplate(input_variables=["human_input"], template=template)
# OPENAI_KEY = "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"
#
# chatgpt_chain = LLMChain(
#     llm=ChatOpenAI(openai_api_key=OPENAI_KEY, temperature=0.4, max_tokens=4000, model_name='gpt-3.5-turbo-16k'),
#     prompt=prompt,
#     verbose=False,
#     memory=ConversationBufferWindowMemory(k=0, ai_prefix='BOT'),
# )
#
# toefl_reading_prompt = "Generate TOEFL-like text of 700-900 words on one of the following topics:\nHistory, Zoology, " \
#                        "Physical Geography,  Biology, Geology, Psychology, Ecology, Architecture, Astronomy, " \
#                        "Sociology, Education, Anthropology, Art, Paleontology"
#
#
# def generate_passage(input_text):
#     output = chatgpt_chain.predict(
#         human_input=toefl_reading_prompt
#     )
#     print(output)
#     return output
#
#
# generate_passage(toefl_reading_prompt)