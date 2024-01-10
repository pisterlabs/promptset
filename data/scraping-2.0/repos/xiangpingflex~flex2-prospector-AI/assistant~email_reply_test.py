# import streamlit as st
# from langchain.document_loaders import JSONLoader
# from langchain.vectorstores import FAISS
# from langchain.embeddings.openai import OpenAIEmbeddings
# from langchain.prompts import PromptTemplate
# from langchain.chat_models import ChatOpenAI
# from langchain.chains import LLMChain
# from dotenv import load_dotenv
# import os
#
# from assistant.common.constant import FINE_TUNED_GPT_4
# from assistant.model.knowledge_base import KnowledgeBase
#
# st.set_page_config(page_title="Customer response generator", page_icon=":bird:")
# load_dotenv()
#
#
# # 1. create knowledge base
# @st.cache_resource
# def create_knowledge_base():
#     print('calling create_knowledge_base')
#     return KnowledgeBase.create_knowledge_base()
#
#
# db = create_knowledge_base()
#
# # def knowledge_base_init():
# #     print('calling knowledge_base_init')
# #     loader = JSONLoader(
# #         file_path="./resource/flex_message.jsonl",
# #         jq_schema='"question: "+.question + " answer: " +.answer',
# #         json_lines=True,
# #     )
# #
# #     documents = loader.load()
# #     embeddings = OpenAIEmbeddings(openai_api_key=os.environ.get("OPEN-API-KEY"))
# #     db = FAISS.from_documents(documents, embeddings)
# #     return db
#
#
# # @st.cache_data
# # def gen_test():
# #     print('calling test')
# #     return "test"
# #
# #
# # a = gen_test()
#
# # 2. Function for similarity search
# def retrieve_info(query):
#     similar_response = db.similarity_search(query, k=3)
#     page_contents_array = [doc.page_content for doc in similar_response]
#     return page_contents_array
#
#
# # 3. Setup LLMChain & prompts
# # llm = ChatOpenAI(
# #     openai_api_key=os.environ.get("OPEN-API-KEY"),
# #     temperature=0.9,
# #     # max_tokens=100,
# #     model="gpt-4-1106-preview",
# # )
# #
# # template = """
# # You are a world class business development representative.
# # I will share a prospect's message with you and you will give me the best answer that
# # I should send to this prospect based on past best practies,
# # and you will follow ALL of the rules below:
# #
# # 1/ Response should be very similar or even identical to the past best practies,
# # in terms of length, ton of voice, logical arguments and other details
# #
# # 2/ If the best practice are irrelevant, then try to mimic the style of the best practice to prospect's message
# #
# # Below is a message I received from the prospect:
# # {message}
# #
# # Here is a list of best practies of how we normally respond to prospect in similar scenarios:
# # {best_practice}
# #
# # Please write the best response in an email format that I should send to this prospect, always ask to schedule a meeting
# # with the prospect politely.
# # """
# #
# # prompt = PromptTemplate(input_variables=["message", "best_practice"], template=template)
# #
# # chain = LLMChain(llm=llm, prompt=prompt)
#
#
# # 4. Retrieval augmented generation
# # def generate_response(message):
# #     best_practice = 'Hello, this is a demo'  # retrieve_info(message)
# #     response = chain.run(message=message, best_practice=best_practice)
# #     return response
#
#
# # 5. Build an app with streamlit
# # def main():
#
#
# st.header("Customer response generator :bird:")
# message = st.text_area("customer message")
#
# if message:
#     st.write("Generating best practice message...")
#
#     result = generate_response(message)
#
#     st.info(result)
#
# # if __name__ == "__main__":
# #     main()
