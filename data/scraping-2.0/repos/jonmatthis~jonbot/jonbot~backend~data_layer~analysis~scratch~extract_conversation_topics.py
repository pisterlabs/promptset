import asyncio
import json
from pathlib import Path
from typing import Dict, Any

from langchain import PromptTemplate, ConversationChain
from langchain.callbacks import StdOutCallbackHandler
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationEntityMemory

from jonbot import logger
from jonbot.backend.data_layer.analysis.get_chats import get_chats
from jonbot.backend.data_layer.analysis.utilities import get_human_ai_message_pairs
from jonbot.backend.data_layer.models.discord_stuff.discord_chat_document import DiscordChatDocument

_DEFAULT_TOPIC_CONVERSATION_TEMPLATE = """

Your job is to examine a conversation between a student and an AI chatbot and extract the topics that they discussed. 

You will be fed in messages in the form of a Human input and the associated AI response.
Context:
{entities}

Current conversation:
{history}
Recent Conversation:
{input}
You:"""

TOPIC_CONVERSATION_TEMPLATE = PromptTemplate(
    input_variables=["entities", "history", "input"],
    template=_DEFAULT_TOPIC_CONVERSATION_TEMPLATE,
)

_DEFAULT_TOPIC_SUMMARIZATION_TEMPLATE = """You are an AI teaching assistant helping a human keep track of topics 
that a student has been discussing with a chatbot. You will be fed in Human Message\AI Response pairs one at a time. 

Update the summary of the topics that have been discussed in this 
conversation in the "Topics" section based on the recent messages exchanged with the human. 
 
If you are writing the summary for the first time, return a single sentence.

The update should only include information gathered from the recent conversation. The Focus should be on understanding 
the the topics that the human was interested in and may want to know more about. 

If there is no new information about the provided topic or the information is not worth noting (not an important or 
relevant fact to remember long-term), return the existing summary unchanged. Focus on topics that are the kinds of 
things that would be discussed in upper-level graduate courses at prestigious universities.

Full conversation history (for context):
+++
{history}
+++

Topics to summarize:
+++
{entity}
+++

Existing summary of {entity}:
+++
{summary}
+++

============================
Recent Conversation:
Current Human Message\AI Response pair:
+++
{input}
+++


Updated summary:"""

TOPIC_SUMMARIZATION_PROMPT = PromptTemplate(
    input_variables=["entity", "summary", "history", "input"],
    template=_DEFAULT_TOPIC_SUMMARIZATION_TEMPLATE,
)

_DEFAULT_TOPIC_EXTRACTION_TEMPLATE = """You are an AI assistant reading the transcript of a conversation between an AI 
and a human. Extract all of the topics  from the  conversation that are relevant to this course;

Only extract topics that are relevant to this course. 

Here is the course description -- you can use this to determine what topics are relevant to this course. This is not 
and exhaustive list of topics, but it should give you a good idea of what kinds of topics are relevant to this course.
 
===
## Course Description
Students will explore the neural basis of natural human behavior in real-world contexts (e.g., [sports], [dance], 
or [everyday-activities]) by investigating the [neural-control] of [full-body] [human-movement]. The course will cover 
[philosophical], [technological], and [scientific] aspects related to the study of [natural-behavior] while emphasizing 
hands-on, project-based learning. Students will use [free-open-source-software], and [artificial-intelligence],
[machine-learning] and [computer-vision] driven tools and methods to record human movement in unconstrained 
environments.

The course promotes interdisciplinary collaboration and introduces modern techniques for decentralized 
[project-management], [AI-assisted-research-techniques], and [Python]-based programming (No prior programming 
experience is required). Students will receive training in the use of AI technology for project management and research 
conduct, including [literature-review], [data-analysis], [data-visualization], and [presentation-of-results]. Students 
will develop valuable skills in planning and executing technology-driven research projects while examining the impact 
of structural inequities on scientific inquiry.

    
## Course Objectives
- Gain exposure to key concepts related to neural control of human movement.
- Apply interdisciplinary approaches when collaborating on complex problems.
- Develop a basic understanding of machine-learning tools for recording human movements.
- Contribute effectively within a team setting towards achieving common goals.
- Acquire valuable skills in data analysis or background research.
===

Return the output as a single comma-separated list, or NONE if there is nothing of note to return (e.g. the user is
 just issuing a greeting or having a simple conversation).

Information will come in the form of Human Message/AI Response pairs. 
EXAMPLE
Conversation history:
Human: how's it going today?
AI: "It's going great! How about you?"
Human: good! I'm curious about the cerebellum. What does it do?
AI: "The cerebellum is a part of the brain that controls movement and balance. It also helps with coordination, 
posture, and speech."
Recent Conversation:
Human: "Good! I'm curious about the cerebellum. What does it do?"
AI: "The cerebellum is a part of the brain that controls movement and balance. It also helps with coordination, 
Output: Cerebellum, Neuroscience, Brain, Biology, Coordination, Movement, Balance
END OF EXAMPLE

EXAMPLE
Conversation history:
Human: how's it going today?
AI: "It's going great! How about you?"
Human: good! I'm curious about the cerebellum. What does it do?
AI: "The cerebellum is a part of the brain that controls movement and balance. It also helps with coordination, 
posture, and speech."
Human: "That's interesting! How an we use motion capture to study the cerebellum?"
AI: "Motion capture is a technology that records the movement of objects or people. It can be used to study the 
cerebellum by recording how it affects a person's movements.
Recent Conversation:
Human: "That's interesting! How an we use motion capture to study the cerebellum?"
AI: AI: "Motion capture is a technology that records the movement of objects or people. It can be used to study the cerebellum by recording how it affects a person's movements. 
Output: Motion capture, Neuroscience, Brain, Technology, Movement
END OF EXAMPLE

Conversation history (for reference only):
{history}
Recent conversation (for extraction):
{input}

Output:"""
TOPIC_EXTRACTION_PROMPT = PromptTemplate(
    input_variables=["history", "input"], template=_DEFAULT_TOPIC_EXTRACTION_TEMPLATE
)


async def extract_conversation_topics(chats: Dict[str, DiscordChatDocument],
                                      save_path: str) -> Dict[str, Any]:
    topics_by_chat = {}
    tasks = []
    try:
        for chat_number, chat_item in enumerate(chats.items()):
            chat_id, chat = chat_item
            if chat_number > 5:
                break
            await extract_topics_from_chat(chat, chat_id, save_path, topics_by_chat)

        await asyncio.gather(*tasks)
    except Exception as e:
        logger.exception(e)
        raise e
    finally:
        return topics_by_chat


async def extract_topics_from_chat(chat: DiscordChatDocument,
                                   chat_id: int,
                                   save_path: str,
                                   topics_by_chat: Dict[str, Any]):
    print(f"\n=====================================\n"
          f"Extracting topics from chat {chat_id}..."
          f"There are {len(chat.messages)} messages in this chat."
          f"\n_____________________________________\n")
    llm = ChatOpenAI(temperature=0,
                     model_name="gpt-3.5-turbo",
                     callbacks=[StdOutCallbackHandler()],
                     streaming=True)
    topic_memory = ConversationEntityMemory(llm=llm,
                                            entity_extraction_prompt=TOPIC_EXTRACTION_PROMPT,
                                            entity_summarization_prompt=TOPIC_SUMMARIZATION_PROMPT,
                                            )
    conversation_chain = ConversationChain(
        llm=llm,
        verbose=True,
        prompt=TOPIC_CONVERSATION_TEMPLATE,
        memory=topic_memory
    )
    human_ai_pairs = get_human_ai_message_pairs(chat)
    topics = []
    for human_message, ai_response in human_ai_pairs:
        # inputs = {"input": human_message}
        # outputs = {"output": ai_response}
        # topic_memory.save_context(inputs=inputs,
        #                           outputs=outputs)
        # topics.append(topic_memory.load_memory_variables(inputs={"input": human_message}, ))
        # topic_memory.save_context(inputs=inputs,
        #                           outputs=outputs)
        input_string = f"Human: {human_message}\nAI: {ai_response}"
        conversation_chain.run(input=input_string)
    json_filename = f"{chat_id}_extracted_topics.json"
    json_path = Path(save_path) / json_filename
    with open(str(json_path), "w", encoding="utf-8") as file:
        json.dump(topic_memory.entity_store.store, file, indent=4)
    print(f"Saved conversation topic memory to {json_path}")
    topics_by_chat[chat_id] = topic_memory.entity_store.store


if __name__ == "__main__":
    database_name = "classbot_database"
    server_id = 1150736235430686720
    chats_in = asyncio.run(get_chats(database_name=database_name,
                                     query={"server_id": server_id}))
    save_path = Path().home() / "syncthing_folders" / "jon_main_syncthing" / "jonbot_data"
    save_path.mkdir(parents=True, exist_ok=True)
    topics_by_chat = asyncio.run(extract_conversation_topics(chats=chats_in,
                                                             save_path=str(save_path)))

    json_filename_out = f"_all_chats_extracted_topics.json"
    json_path_out = Path(save_path) / json_filename_out

    with open(str(json_path_out), "w", encoding="utf-8") as file:
        json.dump(topics_by_chat, file, indent=4)
