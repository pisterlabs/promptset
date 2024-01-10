# from langchain.chains import LLMChain, load_chain, VectorDBQAWithSourcesChain
# from langchain.vectorstores import PineconeStore
# from langchain.prompts import PromptTemplate
# from langchain.callbacks import CallbackManager
# from langchain.llms import OpenAIChat

# CONDENSE_PROMPT = PromptTemplate.fromTemplate(
# 	"Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question. \
# 	Chat History: \
# 	{chat_history} \
# 	Follow Up Input: {question} \
# 	Standalone question:");

# QA_PROMPT = PromptTemplate.fromTemplate(
#   "You are an AI assistant providing helpful advice. You are given the following extracted parts of a long document and a question. Provide a conversational answer based on the context provided. \
# 	You should only provide hyperlinks that reference the context below. Do NOT make up hyperlinks. \
# 	If you can't find the answer in the context below, just say 'Hmm, Im not sure.' Don't try to make up an answer. \
# 	If the question is not related to the context, politely respond that you are tuned to only answer questions that are related to the context. \
# 	Question: {question} \
# 	========= \
# 	{context} \
# 	========= \
# 	Answer in Markdown:"
# );

# def make_chain(vectorstore, on_token_stream=None):
# 		question_generator = LLMChain(
# 				llm=OpenAIChat(temperature=0),
# 				prompt=CONDENSE_PROMPT,
# 		)
# 		doc_chain = load_chain(
# 				OpenAIChat(
# 						temperature=0,
# 						model_name="gpt-3.5-turbo",  # change this to older versions (e.g. gpt-3.5-turbo) if you don't have access to gpt-4
# 						streaming=bool(on_token_stream),
# 						callback_manager=on_token_stream
# 						and CallbackManager.from_handlers(
# 								{
# 										"handle_llm_new_token": lambda token: on_token_stream(token),
# 								}
# 						),
# 				),
# 				prompt=QA_PROMPT,
# 		)

# 		return VectorDBQAWithSourcesChain(
# 				vectorstore=vectorstore,
# 				combine_documents_chain=doc_chain,
# 				question_generator_chain=question_generator,
# 				return_source_documents=True,
# 				k=2,  # number of source documents to return
# 		)
