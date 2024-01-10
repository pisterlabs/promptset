import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage
)

from langchain.prompts import PromptTemplate
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    AIMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.chat_models import ChatOpenAI

# Initialize the ChatOpenAI object
#chat = ChatOpenAI()
chat = None

if "OPENAI_API_KEY" not in st.session_state:
    st.session_state["OPENAI_API_KEY"] = ""
elif st.session_state["OPENAI_API_KEY"] != "":
    chat = ChatOpenAI(openai_api_key=st.session_state["OPENAI_API_KEY"])

if "PINECONE_API_KEY" not in st.session_state:
    st.session_state["PINECONE_API_KEY"] = ""

if "PINECONE_ENVIRONMENT" not in st.session_state:
    st.session_state["PINECONE_ENVIRONMENT"] = ""

st.set_page_config(page_title="Welcome to ASL", layout="wide")

st.title("🤠 Welcome to FinGPT-Columbia")

SystemPrompt1 = '''You are investment advisor, an automated service to ask comprehensive set of survey questions to gather essential information from our clients. 

You first greet the customer.

And then asks clients survey questions. 

You can only ask one question at each time.

You need to ask simple questions, don't be too fussy.

Imagine that your customer base is college students.

Here are some examples of questions, you can generate questions based on these:

What are your main financial goals? (Like paying off student loans, saving for short-term goals, planning a trip, etc.).

Do you want to have any savings or funds for emergency? How much? 

What is your comfort level with investment risk? (very low,low, medium, high,very high)

What is your expected rate of return for your investments?  

Which kind of investment period you are looking for?(1-3 years,3-5 years, over 5 years)

Is your investment plan short-term, medium-term, or long-term?  

Which of the following industries and markets do you have a preference for, such as :Healthcare, Finance, Real Estate, Consumer Discretionary, Consumer Staples, Energy, Industrials, Materials, Communication Services, Utilities, International Markets, Emerging Markets, ESG, AI & Technology, Biotech. 

And what is the preference level for that interest? You can answer: not interested, a little interested, moderately interested, very interested, extremely interested.

You must follow this rule: starting with your second answer, ask the clients questions that relate to the previous client's answers.'''

if 'messages' not in st.session_state:
    st.session_state['messages'] = []

if chat:
    with st.container():
        st.header("FinGPT!!!")

        for message in st.session_state['messages']:

            # '''
            # isinstance() 是 Python 中的一个内置函数,用于判断一个对象是否是一个已知的类型。
            # 在这个代码示例中,它被用来判断存储在 st.session_state['messages'] 中的对象是
            # HumanMessage 类型还是 AIMessage 类型。
            # '''

            if isinstance(message, HumanMessage):
                with st.chat_message('user'):
                    st.markdown(message.content)
            elif isinstance(message, AIMessage):
                with st.chat_message('assistant'):
                    st.markdown(message.content)

        user_input = st.chat_input('Type something...')
        if user_input:
            st.session_state['messages'].append(HumanMessage(content=user_input))
            with st.chat_message('user'):
                st.markdown(user_input)
            ai_message = chat([SystemMessage(content=SystemPrompt1),
                               HumanMessage(content=user_input)])
            st.session_state['messages'].append(ai_message)
            with st.chat_message('assistant'):
                st.markdown(ai_message.content)

else:
    with st.container():
        st.warning("Please set your OpenAI API key in the settings page.")

