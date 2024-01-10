# # https://openai.com/sitemap.xml sitemap을 사용해서 웹 사이트에 존재하는 모든 page의 url을 찾아낼 수 있음
# # playwright: 브라우저 컨트롤을 할 수 있는 패키지(selenium과 비슷)
# # chronium
# # playwright install: 설치 방식이 좀 특이하네
# # playwright를 headless 모드로 실행 -> headlessfks browser process가 내 컴퓨터로부터 시작되는 것을 의미
# # -> 속도가 느려짐
# # 웹 사이트 스크랩을 너무 빠르게하면 차단당할 수 있음... -> Sitemap에서는 1초에 한 번씩 호출 수행
# # metadata 저장도 확인 가능 : metadata={'source': 'https://openai.com/research/weak-to-strong-generalization', 'loc': 'https://openai.com/research/weak-to-strong-generalization', 'lastmod': '2023-12-16T00:32:09.053Z', 'changefreq': 'daily', 'priority': '1.0'})
# # SitemapLoader는 내부적으로 beautiful soup 사용
# # vector store를 만들어서 연관 있는 docu를 검색
# # llm에게 답변의 유용함 평가 요청


# import streamlit as st
# from langchain.document_loaders import AsyncChromiumLoader
# from langchain.document_transformers import Html2TextTransformer
# from langchain.document_loaders import SitemapLoader
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.vectorstores.faiss import FAISS
# from langchain.embeddings import OpenAIEmbeddings
# from langchain.schema.runnable import RunnablePassthrough, RunnableLambda
# from langchain.chat_models import ChatOpenAI
# from langchain.prompts import ChatPromptTemplate



# st.set_page_config(
#     page_title="SiteGPT",
#     page_icon="🐋",
# )

# st.title("SiteGPT")
# st.markdown(
#     """
#     Ask question about the content of a website
    
#     Start by writing the URL of the website on the sidebar
#     """
# )

# # html을 받아서 text로 변환
# html2text_transformer = Html2TextTransformer()


# # 사이드바 입력 칸 생성
# with st.sidebar:
#     url = st.text_input(
#         "Write down a URL", 
#         placeholder="https://example.com",
#         )


# # 사이드바에 url을 입력하면, 해당 페이지의 html을 읽어옴
# # if url:
# #     # async chromium loader
# #     loader = AsyncChromiumLoader([url])
# #     docs = loader.load()
# #     transformed = html2text_transformer.transform_documents(docs)
# #     st.write(docs)

# # SitemapLoader 사용
# # if url:
# #     if ".xml" not in url:
# #         with st.sidebar:
# #             st.error("Please write down a Sitemap URL.")
# #     else:
# #         loader = SitemapLoader(url)
# #         # loader.requests_per_second = 1
# #         docs = loader.load()
# #         st.write(docs)

 
# llm = ChatOpenAI(
#     temperature=0.1,
# )

# answers_prompt = ChatPromptTemplate.from_template(
#     """
#         Using ONLY the following context answer the user's question. If you can't just say you don't know, don't make anything up.
                                                    
#         Then, give a score to the answer between 0 and 5.
#         If the answer answers the user question the score should be high, else it should be low.
#         Make sure to always include the answer's score even if it's 0.
#         Context: {context}
                                                    
#         Examples:
                                                    
#         Question: How far away is the moon?
#         Answer: The moon is 384,400 km away.
#         Score: 5
                                                    
#         Question: How far away is the sun?
#         Answer: I don't know
#         Score: 0
                                                    
#         Your turn!
#         Question: {question}
#     """
# )


# # soup은 beautifule soup object로 된 html 덩어리로 검색이나 삭제 작업 수행 가능
# def parse_page(soup):
#     header = soup.find("header")
#     footer = soup.find("footer")
#     if header:
#         header.decompose()  # decompose: 제거
#     if footer:
#         footer.decompose()
#     return (
#         str(soup.get_text())
#         .replace("\n", " ")
#         .replace("\xa0", " ")
#         .replace("ClosingSearch Submit Blog", "")
#     )
    
    
# # 캐싱되는, url의 텍스트만 가져오는 함수 
# @st.cache_data(show_spinner="Loading website...")
# def load_website(url):
#     # splitter를 정의하여 load_and_split와 함께 사용 가능
#     splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
#         chunk_size=1000,
#         chunk_overlap=200,
#     )

#     loader = SitemapLoader(
#         url,
#         # filter_urls=["http://openai.com/blog/data-par..."],  # 직접적으로 주소를 줄 수도 있고
#         # filter_urls=[r"^(?!.*\/blog\/).*"],  # 정규표현식도 사용 가능(exclude /blog/)
#         # filter_urls=[r"^(.*\/blog\/).*"],  # 정규표현식도 사용 가능(include /blog/)
#         parsing_function=parse_page
#     )
#     loader.requests_per_second = 2
#     # docs = loader.load()
#     docs = loader.load_and_split(text_splitter=splitter)
#     vector_score = FAISS.from_documents(docs,OpenAIEmbeddings())
#     return vector_score.as_retriever()


# choose_prompt = ChatPromptTemplate.from_messages(
#     [
#         (
#             "system",
#             """
#             Use ONLY the following pre-existing answers to answer the user's question.
#             Use the answers that have the highest score (more helpful) and favor the most recent ones.
#             Cite sources and return the sources of the answers as they are, do not change them.
#             Answers: {answers}
#             """,
#         ),
#         ("human", "{question}"),
#     ]
# )


# def choose_answer(inputs):
#     answers = inputs["answers"]
#     question = inputs["question"]
#     choose_chain = choose_prompt | llm
#     condensed = "\n\n".join(
#         f"Answer: {answer['answer']}\nSource:{answer['source']}\nDate:{answer['date']}" for answer in answers
#     )
#     return choose_chain.invoke({
#         "question": question,
#         "answers": condensed
#     })


# def get_answers(inputs):
#     docs = inputs["docs"]
#     question = inputs["question"]
#     answers_chain = answers_prompt | llm
#     # answers = []
#     # for doc in docs:
#     #     result = answers_chain.invoke({
#     #         "question": question,
#     #         "context": doc.page_content
#     #     })
#     #     answers.append(result.content)
#     # st.write(answers)
#     return {
#         "question": question, 
#         "answers": [
#             {
#                 "answer": answers_chain.invoke(
#                     {"question": question, "context": doc.page_content}
#                 ).content,
#                 "source": doc.metadata["source"],
#                 "date": doc.metadata["lastmod"],
#             } for doc in docs
#         ]
#     }


# if url:
#     if ".xml" not in url:
#         with st.sidebar:
#             st.error("Please write down a Sitemap URL.")
#     else:
#         retriever = load_website(url)
#         query = st.text_input("Ask a question to the websie.")
#         if query:
#             chain = {
#                 "docs": retriever, 
#                 "question": RunnablePassthrough(),
#             } | RunnableLambda(get_answers) | RunnableLambda(choose_answer)
            
#             result = chain.invoke(query)
#             st.write(result.content)