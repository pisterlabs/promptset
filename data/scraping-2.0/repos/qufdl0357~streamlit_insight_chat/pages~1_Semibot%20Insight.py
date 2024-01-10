import os
import streamlit as st
from streamlit_agent.clear_results import with_clear_container

from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.memory.chat_message_histories import StreamlitChatMessageHistory
from langchain.callbacks.base import BaseCallbackHandler
from langchain.agents import AgentType
from langchain.agents import initialize_agent, Tool
from langchain.callbacks import StreamlitCallbackHandler
from langchain.chains import RetrievalQA
from langchain.prompts import ChatPromptTemplate
from langchain.vectorstores import Vectara
from langchain.utilities import GoogleSearchAPIWrapper

st.set_page_config(
    page_title="SemiSmart InsightBot", page_icon="🤖", layout="wide"
)

"# 🤖🔗 SemiSmart InsightBot"

# with st.sidebar:
#     user_openai_api_key = st.text_input("OpenAI API Key", key="chatbot_api_key", type="password")

# API_KEY Setting
os.environ["OPENAI_API_KEY"]      = st.secrets.OPENAI_API_KEY

os.environ["VECTARA_CUSTOMER_ID"] = st.secrets.VECTARA_CUSTOMER_ID
os.environ["VECTARA_CORPUS_ID"]   = st.secrets.VECTARA_CORPUS_ID
os.environ["VECTARA_API_KEY"]     = st.secrets.VECTARA_API_KEY

os.environ["GOOGLE_API_KEY"]      = st.secrets.GOOGLE_API_KEY
os.environ["GOOGLE_CSE_ID"]       = st.secrets.GOOGLE_CSE_ID

# Setup credentials in Streamlit
user_openai_api_key = os.getenv("OPENAI_API_KEY")

# Vectara Initialize
vectara = Vectara(
        vectara_customer_id = os.getenv("VECTARA_CUSTOMER_ID"),
        vectara_corpus_id   = os.getenv("VECTARA_CORPUS_ID"),
        vectara_api_key     = os.getenv("VECTARA_API_KEY")
    )

# Setup Keywords
language = ['Semiconductor industry outlook', 'Semiconductor market trends', 'Future of semiconductor technology', 'Semiconductor industry analysis'
            , 'Semiconductor market research', 'Semiconductor market dynamics', 'Semiconductor market challenges', 'Global semiconductor demand'
            , 'Semiconductor market size and share', 'Semiconductor market drivers', 'Semiconductor industry developments', 'Semiconductor market disruptions'
            , 'Semiconductor manufacturing in South Korea', 'South Korea semiconductor exports', 'South Korea semiconductor supply chain'
            , 'Emerging semiconductor technologies in Korea', 'South Korean semiconductor innovations', 'Korean semiconductor trade policies'
            , 'South Korean semiconductor challenges']
selected_keywords = st.multiselect('Select Keyword', language)

# Define retriever
#retriever = vectara.as_retriever(search_type="similarity", search_kwargs={"k": 2, "fetch_k": 4})
retriever = vectara.as_retriever(search_type="similarity", search_kwargs={"k": 2})

if user_openai_api_key:
    openai_api_key = user_openai_api_key
    enable_custom = True
else:
    openai_api_key = "not_supplied"
    enable_custom = False

# Setup memory for contextual conversation
msgs = StreamlitChatMessageHistory()
memory = ConversationBufferMemory(memory_key="chat_history", chat_memory=msgs, return_messages=True)

# Setup LLM 
llm = ChatOpenAI(
    model_name="gpt-3.5-turbo", openai_api_key=openai_api_key, temperature=0, streaming=True
)

# Create KnowledgeBase_Prompt
knowledgeBase_template = """
SYSTEM
You are an expert researcher and writer, tasked with answering any question.

Generate a comprehensive and informative, yet concise answer of 250 words or less for the given question based solely on the provided search results (URL and content).
You must only use information from the provided search results. Use an unbiased and journalistic tone. Combine search results together into a coherent answer.
Do not repeat text. Cite search results using [${{number}}] notation. Only cite the most relevant results that answer the question accurately.
Place these citations at the end of the sentence or paragraph that reference them - do not put them all at the end.
If different results refer to different entities within the same name, write separate answers for each entity.
If you want to cite multiple results for the same sentence, format it as `[${{number1}}] [${{number2}}]`.
However, you should NEVER do this with the same number - if you want to cite `number1` multiple times for a sentence, only do `[${{number1}}]` not `[${{number1}}] [${{number1}}]`

You should use bullet points in your answer for readability. Put citations where they apply rather than putting them all at the end.
If there is nothing in the context relevant to the question at hand, just say "Hmm, I'm not sure." Don't try to make up an answer.
Anything between the following `context` html blocks is retrieved from a knowledge bank, not part of the conversation with the user.
You must answer in Korean.

<context>
    {context}
<context/>

HUMAN
{question}
  """
knowledgeBase_prompt = ChatPromptTemplate.from_template(knowledgeBase_template)

# retrieval qa chain
knowledgeBase_qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    chain_type_kwargs={"prompt": knowledgeBase_prompt}
)

# search = GoogleSearchAPIWrapper()

tools = [
     Tool(
        name='Knowledge Base',
        func=knowledgeBase_qa.run,
        description=(
            'use this tool when answering general knowledge queries to get '#tool description 수정 필요
            'more information about the topic'
        )
    ),
    # Tool(
    #     name="Google Search",
    #     func=search.run,
    #     description="Search Google for recent results.",#tool description 수정 필요
    #  )
]

# Initialize agent
mrkl = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True,memory=memory,handle_parsing_errors=True)

with st.form(key="form"):
    user_input = st.text_input("Or, ask your own question")
    submit_clicked = st.form_submit_button("Submit Question")

output_container = st.empty()

if with_clear_container(submit_clicked):
    output_container = output_container.container()
    output_container.chat_message("user").write(user_input)

    answer_container = output_container.chat_message("assistant", avatar="🤖")
    st_callback = StreamlitCallbackHandler(answer_container)

    # If we've saved this question, play it back instead of actually running LangChain
    # (so that we don't exhaust our API calls unnecessarily)
    answer = mrkl.run(user_input, callbacks=[st_callback])
    
    keyword_string = ' '.join(selected_keywords)
    filters = f"doc.keyword = '{keyword_string}'"

    found_docs = vectara.similarity_search(
    user_input, k=7, n_sentence_context=0, filter=filters
    )

    # answer_container.subheader(f':rainbow[{answer}]')
    answer_container.subheader(answer)



    for i, doc in enumerate(found_docs):
        expander = st.expander(f"Resource {i + 1}")
        expander.markdown(f"Page Content: ***{doc.page_content}***")
        expander.markdown(f":blue[Metadata:{doc.metadata}]")