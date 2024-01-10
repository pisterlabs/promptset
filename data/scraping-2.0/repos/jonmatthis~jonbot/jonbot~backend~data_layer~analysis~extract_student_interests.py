import asyncio
import json
from pathlib import Path
from typing import Dict, Any, List

from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.text_splitter import CharacterTextSplitter

from jonbot.backend.data_layer.database.mongo_database import MongoDatabaseManager


def split_text_into_chunks(text: str,
                           model: str,
                           chunk_size: int,
                           chunk_overlap_ratio: float = .1) -> List[str]:
    text_splitter = CharacterTextSplitter.from_tiktoken_encoder(
        model_name=model,
        chunk_size=chunk_size,
        chunk_overlap=chunk_size * chunk_overlap_ratio
    )
    texts = text_splitter.split_text(text)
    return texts


SIMPLE_CONVERSATION_SUMMARIZER_PROMPT_TEMPLATE = """
You are an AI teaching assistant for a class called "The Neural Control of Real World Human Movement", analyzing text 
conversations between students and an AI in an effort to understand the interests of the students as well as the general
 intellectual landscape of this class.

Here is a description of this class:

=====
COURSE DESCRIPTION

Students will explore the neural basis of natural human behavior in real-world contexts (e.g., sports, dance, or 
everyday-activities) by investigating the neural-control of full-body human-movement. The course will cover 
philosophical, technological, and scientific aspects related to the study of natural-behavior while emphasizing  
ands-on, project-based learning. Students will use free-open-source-software, and artificial-intelligence, 
machine-learning and computer-vision driven tools and methods to record human movement in unconstrained environments. 
======

Below is a conversation between a student in this class and an AI teaching assistant.

Your job is to summarize this conversation in a way that captures the most important information in the conversation.

+++
{text}
+++
"""

EYE_TRACKING_CONTEXT_DESCRIPTION = """
a channel where the students were asked to provide information, research, and background related to the topic of
vision, eye tracking, and oculomotor control.
"""

TOPIC_EXTRACTOR_PROMPT_TEMPLATE = """
You are an AI teaching assistant for a class called "The Neural Control of Real World Human Movement", analyzing text 
conversations between students and an AI. 


Here is a description of this class:

=====
COURSE DESCRIPTION

Students will explore the neural basis of natural human behavior in real-world contexts (e.g., sports, dance, or 
everyday-activities) by investigating the neural-control of full-body human-movement. The course will cover 
philosophical, technological, and scientific aspects related to the study of natural-behavior while emphasizing  
ands-on, project-based learning. Students will use free-open-source-software, and artificial-intelligence, 
machine-learning and computer-vision driven tools and methods to record human movement in unconstrained environments. 
======

Below is a conversation between a student in the class and an AI teaching assistant.

Your job is to extract tags for the course-relevant topics that are discussed in this conversation.

Your response should of a comma separated list of topics wrapped in [[double brackets]] like this: 


```markdown
SUMMARY: 
- Main point 1
- Main point 2

TOPICS:
[[Topic Name]], [[Another topic name]], [[Yet another topic name]]

```


Here is the conversation (between the  +++ symbols +++):

+++
{text}
+++


Remember: 


Your response should of a comma separated list of topics wrapped in [[double brackets]] like this: 


```markdown
SUMMARY: 
- Main point 1
- Main point 2

TOPICS:
[[Topic Name]], [[Another topic name]], [[Yet another topic name]]

```


"""

TOPIC_ORGANIZER_PROMPT_TEMPLATE = """
You are an AI teaching assistant for a class called "The Neural Control of Real World Human Movement".

Here is a description of this class:

=====
COURSE DESCRIPTION

Students will explore the neural basis of natural human behavior in real-world contexts (e.g., sports, dance, or 
everyday-activities) by investigating the neural-control of full-body human-movement. The course will cover 
philosophical, technological, and scientific aspects related to the study of natural-behavior while emphasizing  
ands-on, project-based learning. Students will use free-open-source-software, and artificial-intelligence, 
machine-learning and computer-vision driven tools and methods to record human movement in unconstrained environments. 
======


Your job is to organize and condense this long list into a single coherent list of summaries and topics that represent the 
topics that the student has discussed with the bot acorss all of their conversations into a single, concise summary and outline of topics that the student is 
most interested in (based on the things they talk about with the bot).


Your response should of a comma separated list of topics wrapped in [[double brackets]] like this: 

```markdown
SUMMARY: 
- Main point 1
- Main point 2

TOPICS:
[[Topic Name]], [[Another topic name]], [[Yet another topic name]]

```

Here is the list of topics that you are helping to organize (between the +++ symbols +++):

+++
{text}
+++

Remember:
Your response should of a comma separated list of topics wrapped in [[double brackets]] like this: 

```markdown
SUMMARY: 
- Main point 1
- Main point 2

TOPICS:
[[Topic Name]], [[Another topic name]], [[Yet another topic name]]

```

BE CONCISE AND DO NOT MAKE THINGS UP
Thank you! 

"""


def create_simple_summary_chain():
    llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo-16k")
    prompt = ChatPromptTemplate.from_template(SIMPLE_CONVERSATION_SUMMARIZER_PROMPT_TEMPLATE)
    chain = prompt | llm
    return chain


def create_topic_extractor_chain():
    llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo-16k")
    prompt = ChatPromptTemplate.from_template(TOPIC_EXTRACTOR_PROMPT_TEMPLATE)

    chain = prompt | llm
    return chain


def create_topic_organizer_chain():
    llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo-16k")
    prompt = ChatPromptTemplate.from_template(TOPIC_ORGANIZER_PROMPT_TEMPLATE)

    chain = prompt | llm
    return chain


async def upsert_chats(chats_by_id: Dict[str, Dict[str, Any]],
                       database_name: str,
                       collection_name: str,
                       mongo_database_manager: MongoDatabaseManager) -> bool:
    entries = []
    for chat_id, chat in chats_by_id.items():
        entries.append({"query": chat["query"],
                        "data": chat})
    await mongo_database_manager.upsert_many(database_name=database_name,
                                             collection_name=collection_name,
                                             entries=entries)


def get_chats_per_student(chats: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    student_ids = []
    for chat in chats:
        for couplet in chat["couplets"]:
            if couplet["human_message"]:
                human_id = couplet["human_message"]["author_id"]["$numberLong"]
                if human_id not in student_ids:
                    student_ids.append(human_id)

    chats_by_student_id = {}
    for student_id in student_ids:
        chats_by_student_id[student_id] = []
        for chat in chats:
            if chat["channel_id"]["$numberLong"] == "1150790751907762256":  # skip "playground" chats
                continue
            for couplet in chat["couplets"]:
                if couplet["human_message"]:
                    human_id = couplet["human_message"]["author_id"]["$numberLong"]
                    if human_id == student_id:
                        if chat not in chats_by_student_id[student_id]:
                            chats_by_student_id[student_id].append(chat)

    return chats_by_student_id


async def extract_student_interests(chats: List[Dict[str, Any]],
                                    save_directory: Path):
    print("\n\n========================\n\n")
    print(f"Analyzing {len(chats)} chats...")
    print("\n\n========================\n\n")

    topic_extractor_chain = create_topic_extractor_chain()
    topic_organizer_chain = create_topic_organizer_chain()

    chats_by_student_id = get_chats_per_student(chats=chats)

    # randomise the order of the students
    student_ids = list(chats_by_student_id.keys())
    import random
    random.shuffle(student_ids)
    chats_by_student_id = {student_id: chats_by_student_id[student_id] for student_id in student_ids}
    topics_by_student_id = {}
    for student_id, chats in chats_by_student_id.items():
        print(f"Analyzing {len(chats)} chats for student: {student_id}")
        topics_by_student_id[student_id] = {}
        # Extract topics from chats
        text_inputs = [{"text": chat["as_text"]} for chat in chats]

        topics_results = await topic_extractor_chain.abatch(inputs=text_inputs)
        topics_str = ""
        for result in topics_results:
            topics_str += "\n\n========================\n\n"
            topics_str += "\n\nNEW CHAT\n\n"
            topics_str += result.content
            topics_str += "\n\n________________________\n\n"

        print("\n\n========================\n\n")
        print("Extracted topics from chats:\n\n")
        print(topics_str)
        print("\n\n========================\n\n")

        # save to markdown file
        topics_md_path = Path(save_directory / "all_results" / f"student_{student_id}_interests_all.md")
        topics_md_path.parent.mkdir(exist_ok=True, parents=True)
        with open(topics_md_path, "w", encoding="utf-8") as file:
            file.write(topics_str)

        # Organize extracted topics into a hierarchical outline
        organized_topics_results = topic_organizer_chain.invoke({"text": topics_str})

        organized_topics_str = organized_topics_results.content

        print("\n\n========================\n\n")
        print(f"Organized topics from chats with student {student_id}:\n\n")
        print(organized_topics_str)
        print("\n\n========================\n\n")

        # save to markdown file
        organized_topics_md_path = Path(
            save_directory / "organized_results" / f"student_{student_id}_interests_organized.md")
        organized_topics_md_path.parent.mkdir(exist_ok=True, parents=True)
        with open(organized_topics_md_path, "w", encoding="utf-8") as file:
            file.write(organized_topics_str)

        topics_by_student_id[student_id] = organized_topics_str

    all_student_topics_str = ""
    for student_id, topics_str in topics_by_student_id.items():
        all_student_topics_str += f"\n\n========================\n\n"
        all_student_topics_str += f"\n\nSTUDENT {student_id}\n\n"
        all_student_topics_str += topics_str
        all_student_topics_str += "\n\n________________________\n\n"

    all_student_topics_md_path = Path(save_directory / "all_students" / f"all_students_interests_all.md")
    all_student_topics_md_path.parent.mkdir(exist_ok=True, parents=True)

    with open(all_student_topics_md_path, "w", encoding="utf-8") as file:
        file.write(all_student_topics_str)

    organized_topics_results = topic_organizer_chain.invoke({"text": all_student_topics_str})

    organized_topics_str = organized_topics_results.content

    print("\n\n========================\n\n")
    print(f"Organized topics from chats with all students:\n\n")
    print(organized_topics_str)
    print("\n\n========================\n\n")

    organized_topics_md_path = Path(
        save_directory / "all_students" / f"all_students_interests_organized.md")
    organized_topics_md_path.parent.mkdir(exist_ok=True, parents=True)
    with open(organized_topics_md_path, "w", encoding="utf-8") as file:
        file.write(organized_topics_str)


if __name__ == "__main__":
    json_path = Path(
        r"C:\Users\jonma\syncthing_folders\jon_main_syncthing\jonbot_data\classbot_database\classbot_database.chats_2023-11-14.json"
    )
    with open(json_path, "r", encoding="utf-8") as file:
        chats_from_file = json.load(file)

    save_directory = Path(
        r"C:\Users\jonma\syncthing_folders\jon_main_syncthing\jonbot_data\classbot_database\student_interests")
    save_directory.mkdir(exist_ok=True, parents=True)

    asyncio.run(extract_student_interests(chats=chats_from_file,
                                          save_directory=save_directory)
                )
