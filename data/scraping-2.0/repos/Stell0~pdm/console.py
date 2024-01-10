from typing import List, Dict, Callable
from langchain.chat_models import ChatOpenAI
from langchain.schema import (
    HumanMessage,
    SystemMessage,
	AIMessage
)
from langchain.llms import OpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
import tiktoken

def make_context(question: str, history: str) -> str:
	context_template = '''You are a tech support specialist, answer the customer question in his own language using following documents as reference:\n\n{documents}\n\n''' 

	embeddings = OpenAIEmbeddings()
	from db import DB
	db = DB()

	doc_with_score = db.vectorstore.similarity_search_with_score(question, k=5)

	# get embeddings for history
	if history != "":
		context_template += "\n\nPrevious answer was:\n\n{history}"
		doc_with_score = doc_with_score + db.vectorstore.similarity_search_with_score(history, k=5)
	context_template += '''Customer question:"{question}"'''
	# order documents by score
	doc_with_score = sorted(doc_with_score, key=lambda x: x[1], reverse=True)

	# remove duplicates
	unique_docs = []
	unique_scores = []
	print('#'*30)
	for doc, score in doc_with_score:
		if score not in unique_scores and score > 0.05:
			unique_docs.append(doc)
			unique_scores.append(score)
			# print document source with score
			print('Source: '+doc.metadata["source"]+" Score: "+str(score))
	
	print('#'*30+"\n")

	chat = ChatOpenAI(temperature=0)
	encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
	# summarize and concatenate summaries of all documents
	system_message_content="I want you to remove from this text non-essential information without translating or altering other content"
	system_message_token_length=len(encoding.encode(system_message_content))
	documents = ""
	for doc in unique_docs:
		# get token length of current document
		doc_token_length = len(encoding.encode(doc.page_content))
		# get token length of all documents
		documents_token_length = len(encoding.encode(documents))
		# if there are enough tokens left, add document to documents
		if doc_token_length + documents_token_length + system_message_token_length < 3800:
			documents += doc.page_content+"\n\n"
		
	summary = chat(
			[
				HumanMessage(content=doc.page_content),
				SystemMessage(content=system_message_content),
			]
		).content
	
	context = context_template.format(documents=summary, question=question, history=history)
	return context

from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
chat = ChatOpenAI(streaming=True, callbacks=[StreamingStdOutCallbackHandler()], temperature=0) # type: ignore
resp = "Hi! I'm your personal data manager. Ask me something about your data.\n"
messages = []
content = ""
human_input = input(resp)
while True:
	messages.append(HumanMessage(content="question: "+human_input))
	context = make_context(human_input,content)
	messages.append(SystemMessage(content=context))
	out = chat(messages)
	content = out.content
	messages.append(AIMessage(content=content))
	print("\n")
	while True:
		human_input = input("\n")
		if human_input != "":
			break
	
