from langchain.llms import OpenAI
from langchain.chains import LLMChain
from langchain.agents import load_tools
from langchain.agents import initialize_agent
from langchain.agents import AgentType
from langchain.agents.load_tools import get_all_tool_names
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import YoutubeLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter


import langchain


def langchain_simple(prompt):
    llm = OpenAI(model_name="text-davinci-003")
    print(llm(prompt))


def langchain_prompt():
    llm = OpenAI(model_name="text-davinci-003")

    # 翻译模板
    translation_template = langchain.prompts.PromptTemplate(
        input_variables=["en_word"],
        template="Translate word '{en_word}' into Chinese",
    )

    # 音标模板
    phonetic_template = langchain.prompts.PromptTemplate(
        input_variables=["en_word"],
        template="Provide the phonetic transcription of the word '{en_word}' in International Phonetic Alphabet",
    )

    # 创建翻译链
    translation_chain = LLMChain(llm=llm, prompt=translation_template)
    translation_result = translation_chain.run("punk")

    # 创建音标链
    phonetic_chain = LLMChain(llm=llm, prompt=phonetic_template)
    phonetic_result = phonetic_chain.run(translation_result)

    print(phonetic_result)


def langchain_agent():
    llm = ChatOpenAI()

    get_all_tool_names()
    tools = load_tools(["requests_all"], llm=llm)

    # Finally, let's initialize an agent with the tools, the language model, and the type of agent we want to use.
    agent = initialize_agent(
        tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True
    )

    # Now let's test it out!
    result = agent.run(
        "帮我概括一下这篇地址中的内容：https://platform.openai.com/docs/api-reference/assistants/deleteAssistantFile"
    )
    print(result)


def langchain_chat():
    g_llm = OpenAI()
    g_conversation = langchain.chains.ConversationChain(llm=g_llm, verbose=True)
    while True:
        user_input = input("user: ")
        if user_input == 'exit':
            break
        output = g_conversation.predict(input=user_input)
        print('assistant:', output)


def download_youtube(video_url):
    loader = YoutubeLoader.from_youtube_url(video_url)
    transcript = loader.load()
    print(transcript)

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=100)
    docs = text_splitter.split_documents(transcript)


# langchain_chat()

# langchain_state()

# langchain_agent()

# langchain_simple('example的缩写是什么')

# langchain_prompt()

download_youtube("https://www.youtube.com/watch?v=UgX5lgv4uVM&list=PLAcBKcB4AjbGq3ksftZo5Vso03uWiUnd2&index=7&t=836s")
