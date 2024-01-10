from langchain.chains import ConversationChain
from langchain.memory import ConversationSummaryBufferMemory
from langchain.llms import OpenAI

llm = OpenAI(temperature=0)

convo_memory = ConversationSummaryBufferMemory(llm=llm, max_token_limit=10, return_messages=True)
convo_memory.save_context({"input": "hi"}, {"output": "whats up"})
convo_memory.save_context({"input": "what is a dog?"}, {"output": "A  dog is a domesticated mammal of the family Canidae, typically kept as a pet or for work or hunting. Dogs are often referred to as \"man's best friend\" due to their loyalty and companionship."})

conversation_with_summary = ConversationChain(
		llm=llm,
		memory=convo_memory,
		verbose=True,
	)

user_input = ""
while user_input != "stop":
	# thing = memory.load_memory_variables({})

# print(thing)

# messages = memory.chat_memory.messages
# previous_summary = ""
# thing2 = memory.predict_new_summary(messages, previous_summary)

# print(thing2)

	user_input = input("")

	thing = conversation_with_summary.predict(input=user_input)

	print(thing)


