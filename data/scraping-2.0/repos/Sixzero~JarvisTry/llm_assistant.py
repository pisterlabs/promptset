#%%
from langchain_core.runnables import RunnablePassthrough

from llm_helper import get_llm_model
from utils import append_jsonl


# llm = get_llm_model("/llms/gguf/openhermes-2.5-neural-chat-v3-3-slerp.Q6_K.gguf")
# llm = get_llm_model("/llms/gguf/openhermes-2-mistral-7b.Q4_K_M.gguf")
llm = get_llm_model("/llms/gguf/dolphin-2.6-mistral-7b.Q5_K_M.gguf")




def is_AI_called_Jarvis(text, valid=None):
	res = llm.invoke(text)
	if valid is not None:
		if ((valid == True and res.startswith("No")) or (valid == False and res.startswith("Yes"))):
			print(f'{valid}:', res)
		else:
			print("OK:",res)
		
	return res.startswith("Yes")

def is_AI_called(text, valid=None):
	res = llm.invoke(f"""Message: '{text}.'\n 
You are an assistant. You can be referred with Jarvis.  
Your task is to decide whether you were referenced in a message. Answer with 'Yes' or 'No' """ +
# " Please reason why you think you were mentioned. " +
					 "\n")
	if valid:  #  and not res.startswith(valid)
		print(f'{valid}:', res)
		pass
	return res.startswith("Yes")

def get_me_diff_llm(last_text, text):
	res = llm.invoke(f"""Original: '{last_text}'\n New text: {text}. \n
The texts has a part which is similar, but I will only need the part after that similar part? The two texts are from speech, so they might have typos, which you can fix of course. Only write me new words after this sam part word by word in one line.\n """)
	return res

def find_same_from_end(last_text, text):
	ltxt = last_text.lower()
	ntxt = text.lower()
	# replace all dots with one replace,
	ltxt = ltxt.replace(".", " ")
	ntxt = ntxt.replace(".", " ")
	# split by space
	ltxt = ltxt.split(" ")
	ntxt = ntxt.split(" ")
	# Start from the end of the shorter text
	min_length = min(len(ltxt), len(ntxt))
	min_cut = 9999
	min_cut_i = 9999
	# Find the index where texts start to differ
	for i in range(1, min_length + 1):
		if i>=min_cut:
			print('i>=min_cut:', i, min_cut)
			break
		for j,v in enumerate(reversed(ntxt)):
			if j>min_cut:
				break
			if ltxt[-i] == v and j<min_cut:
				min_cut = j + 1
				min_cut_i = i
				break

	# Extract the differing parts
	if min_cut == 0:
		print("LT:",last_text)
		print("TX:",text)
		return "", ""
	last_text_diff = last_text.split(" ")[-min_cut_i+1:]
	text_diff = text.split(" ")[-min_cut-1:]
	last_text_diff = ' '.join(last_text_diff)
	text_diff = ' '.join(text_diff)
	# print('difference_in_last_text:', last_text_diff)
	# print('difference_in_text:', text_diff)

	return last_text_diff, text_diff

from time import sleep, time
from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from tts_helper import add_text2queue, init_talk_processor, playing_until
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain.prompts import HumanMessagePromptTemplate, ChatPromptTemplate
from RAG_db import get_retriever, context_idx_applier, perform_actions

job_messages = ChatPromptTemplate.from_messages([
	("system", "You are ChatGPT an AI assistant, who is responsible for database transactions. Your task is to decide, whether you were asked to change something in the database or not. If you were say: 'I was asked.' if you were not say: 'I was not asked'. In the next line you can write the action we should do, we either ADD new record if someone asks to take a not of something and it is not in the context or MODIFY the database if someone asks you to change comething existing in the context. You have 2 actions: ADD or MODIFY."), # 4->3 , DELETE
	("human", """Context: {context} \nMessage: {question}\nYour task is to decide wether we need to change a context. If we need to add new context write: ADD('what we need to add as a string') If we need to modify existing context, write: MODIFY(index, 'new content'), If there is multiple modification needed, write them in separate lines, if there is nothing to change write NONE."""), #  if we need to delete, write: DELETE(index)
])
talk_template = ChatPromptTemplate.from_messages([
	("system", "You are the ChatGPT AI assistant, also known as Jarvis, Jervis, Józsi, József, or ChatGPT. You are capable of understanding and responding in both English and Hungarian. If you think you were mentioned, write 'Yes,' and answer, otherwise write 'Silent' and explain. Always answer in the language you were asked."
	),
	("human", """Context:\n```\n{context}\n```\n\nQuestion:\n```\n{question}\n```\n\nDetermine if your name is mentioned in the question. If it is, answer the question using the provided context. 
	The Question is a speech recognized text, so you can suppose some words sound similar to other, so you don't have to take every word as it is written in it. There can be other people talking in the background which can lead to not good speech recognition.
	Try to only answer the last question in the sentence to which you can respond, if you have been mentioned and you don't answer the question, the people will be sad, and we don't want people to be sad, so please try to do your best! If not, respond with 'Silent' and explain the reason for your silence."""),
	# ("system", "You are an AI assistant who can be referred to by Jarvis, who helps people in Hungary. Try to focuse on important things. Only answer if you have been mentioned, otherwise write: 'Silent'."),
	# ("human", """The context: {context} \nThe question: {question}"""),
])

def stream_chatgpt(question, mock=False):
	append_jsonl('data/talks2jarvis.raw.jsonl', {'prompt': question})
	init_talk_processor()
	retriever = get_retriever()
	print('prompt:', question)

	model = ChatOpenAI(model="gpt-4-1106-preview", temperature=0.0)
	# model = ChatOpenAI(model="gpt-3.5-turbo-1106", temperature=0.0)
	oai_job = {"context": retriever, "question": RunnablePassthrough()} | context_idx_applier | job_messages | model
	oai_chat = {"context": retriever, "question": RunnablePassthrough()} | talk_template | model
	# oai_chat = ChatOpenAI(model="gpt-4-1106-preview", temperature=0.1)
	
	combined = RunnableParallel(task=oai_job, chat=oai_chat)
	task = ""
	msg = ""
	is_silenced = True
	for msg_dict in combined.stream(question):
	# for chunk in oai_chat.stream(question):
		if 'task' in msg_dict:
			task += msg_dict['task'].content
			continue
		chunk = msg_dict['chat']
		print(chunk.content, end="", flush=True)
		if not is_silenced:
			add_text2queue(chunk.content)
		msg += chunk.content
		if is_silenced and not "Silent".startswith(msg[:6]):
			add_text2queue(msg)
			is_silenced = False
	print()
	print('MSG:', msg)
	print('TASK:', task)
	perform_actions(task, mock)
	return msg
def stream_chatgpt_single(question):
	init_talk_processor()
	retriever = get_retriever()
	print('prompt:', question)

	messages = ChatPromptTemplate.from_messages([
		("system", """You are ChatGPT an AI assistant, who is working in a STEP based process.
	 On the first line you can write 'STEP 1:' and then in next line you have 3 options to write:
	 'CASE 1: I was asked to answer.': If you have been asked to answer a question based on context or you just know the answer. Always answer in the language you were asked
	 'CASE 2: I was asked to change context.': If you have been asked to take a note of a new context, because it is not listed, or if you have been asked to modify an existing context.
	 'CASE 3: I was not asked.': If you were not asked. In this case there is no STEP 2 and STEP 3.
	 After this write: 'STEP 2:'
	 If we are in CASE 1, then answer the question based on context or based on your knowledge. 
	 If we are in CASE 2 then In the next line you can write the action we should perform. You have 2 actions: ADD or MODIFY. we either ADD new record if someone asks to take a note of something and it is not in the context or MODIFY the database if someone asks you to change something existing in the context. 
	 To add new thing to context, if we think it is new information: ADD('The context value that is new.') 
	 In case we need to modify existing context, write: MODIFY(index, 'The new context for the specific context value.')
	 The next line write 'STEP 3:', and then if we are in CASE 2 then write, """), # 4->3 , DELETE
		("human", """Context: {context}\nMessage: {question}\nPlease don't interact with to the context, that is your past memory, and now you can start with writing STEP 1:"""), #  if we need to delete, write: DELETE(index)
	])
	model = ChatOpenAI(model="gpt-4-1106-preview", temperature=0.0)
	model = ChatOpenAI(model="gpt-3.5-turbo-1106", temperature=0.0)
	oai_chat = {"context": retriever, "question": RunnablePassthrough()} | context_idx_applier | messages | model
	msg = ""
	is_silenced = True
	for chunk in oai_chat.stream(question):
		print(chunk.content, end="", flush=True)
		if not is_silenced:
			add_text2queue(chunk.content)
		msg += chunk.content
		if is_silenced and not "Silent".startswith(msg[:6]):
			add_text2queue(msg)
			is_silenced = False
	print('MSG:', msg)
	print()
	return msg
	
	
if __name__ == "__main__":

	retriever = get_retriever()
	oai_job = {"context": retriever, "question": RunnablePassthrough()} | context_idx_applier | job_messages | ChatOpenAI(model="gpt-3.5-turbo-1106", temperature=0.0)
	# LT = "Amikor ugyanaz, akkor valami tud tényleg nem stimmel. Úgyhogy most már tényleg erre kell, hogy mi a probléma."
	# TX = "Amikor ugyanaz, akkor valami tud tényleg nem stimmel. Úgyhogy most már tényleg erre kell, hogy mi a probléma. Egyáltalán."
	# print(find_same_from_end(LT, TX))
	# LT = "Amikor ugyanaz, akkor valami tud tényleg nem stimmel. Úgyhogy most már tényleg erre kell, hogy mi a probléma. Egyáltalán."
	# TX = "Amikor ugyanaz, akkor valami tud tényleg nem stimmel. Úgyhogy most már tényleg erre kell, hogy mi a probléma. Egyáltalán nem ugyanaz a kettő."
	# print(find_same_from_end(LT, TX))
	# LT = "Amikor ugyanaz, akkor valami tuti nem stimmel. Úgyhogy most már tényleg erre kell, hogy mi a probléma? Egyáltalán nem ugyanaz a kettő, amit mondod, meg amit mondjál. De, nem!"
	# TX = "Amikor ugyanaz, akkor valami tud tényleg nem stimmel. Úgyhogy most már tényleg erre kell, hogy mi a probléma. Egyáltalán nem ugyanaz a kettő, amit mondott, meg amit mondja. De, nem! Tehát akkor..."
	# print(find_same_from_end(LT, TX))
	# stream_chatgpt("Milyen az időjárás?")
	# stream_chatgpt("Milyen az időjárás, Jarvis?")
	# stream_chatgpt("Jarvis, ki Tomi barátnője?")
	# stream_chatgpt("Jarvis, mit tudsz Zsófiról?")
	pass
	# res = oai_job.invoke("Jarvis, please note that I need to buy milk.")
	# print('res:', res.content)
	# res = oai_job.invoke("Jarvis, légyszi jegyezd fel, hogy vennem kell tejet.")
	# print('res:', res.content)
	# res = oai_job.invoke("Jarvis, légyszi módosítsd Zsófi szülinapját 1990-ről 1991-re.")
	# res = oai_job.invoke("Jarvis, légyszi módosítsd Zsófijnak az elírását 'Zsófinak'.")
	# print('res:', res.content)
	# res = oai_job.invoke("Jarvis, ki Tomi barátnője?'.")
	# perform_actions(res)
	# print('res:', res)

# %%
