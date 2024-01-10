from langchain.chat_models import ChatOpenAI
from langchain.prompts import (
    ChatPromptTemplate,
    PromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)    
from langchain.text_splitter import RecursiveCharacterTextSplitter
from datetime import datetime
import re

def format_to_seconds(timestamp_str):
    dt = datetime.strptime(timestamp_str, '%H:%M:%S,%f')
    return str(dt.hour * 3600 + dt.minute * 60 + dt.second + dt.microsecond / 1000000)

def createTopics(srt_content):
    print("Getting Topics...")
    print("Splitting into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 2500,
        chunk_overlap  = 200,
        length_function = len,
    )

    texts = text_splitter.create_documents([srt_content])

    print("Total chucks -> " + str(len(texts)))

    llm = ChatOpenAI(model_name="gpt-3.5-turbo")

    topic_list = []

    print("Extracting topics from all chunks...")

    system_message_prompt = SystemMessagePromptTemplate.from_template("You are a helpful, professional and concise AI that provides good topics based on the audio without losing context.")

    for srt_content in texts:
        promptTemplate = PromptTemplate(
            template = """
            Given the following srt content of an audio give me the max of 10 topics with timestamps and a brief summary format of each part of the audio.
            All non-relevant topic should be removed.

            srt_content: {srt_content}

            The output should be in the following format "\n":
            Topic: (topic)
            Summary: (summary)
            Timestamp: (timestamp)
            """,
            input_variables=["srt_content"]
        )

        human_message_prompt = HumanMessagePromptTemplate(prompt=promptTemplate)

        chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])

        chat_messages = chat_prompt.format_prompt(srt_content=srt_content).to_messages()

        result = llm(chat_messages)
        topic_list.append(result.content)

    print("Running merge of topics...")

    promptTemplate = PromptTemplate(
        template = """
        Given the list of topics:

        topic_list: {topic_list}

        Merge them all together the topics together to make a maximum of 10.
        Topics must be ordered by timestamp.

        The output should be in the following format "\n":
        
        (topic_number).
        Topic: (topic)
        Summary: (summary)
        Timestamp: (timestamp)
        """,
        input_variables=["topic_list"]
    )

    system_message_prompt = SystemMessagePromptTemplate.from_template("You are a helpful, professional and concise AI that provides good merged topics based on a list of topics")
    human_message_prompt = HumanMessagePromptTemplate(prompt=promptTemplate)

    chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])

    chat_messages = chat_prompt.format_prompt(topic_list=topic_list).to_messages()

    result = llm(chat_messages)
    data = result.content

    timestamps = re.findall('Timestamp:\s(\d{2}:\d{2}:\d{2},\d{3})\s-\s\d{2}:\d{2}', data)
    topics_list = re.findall('Topic:\s(.*?)\n', data)

    formatted_timestamps_seconds = [format_to_seconds(ts) for ts in timestamps]
    merged_data = []

    for i in range(len(timestamps)):
        merged_element = f"{topics_list[i]} -> {formatted_timestamps_seconds[i]}"
        merged_data.append(merged_element)

    print("Done!")
    
    return (data, merged_data)