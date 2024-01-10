from langchain.vectorstores import FAISS
from langchain.retrievers import TimeWeightedVectorStoreRetriever
from langchain.embeddings import OpenAIEmbeddings
from langchain.docstore import InMemoryDocstore
from langchain.chat_models import ChatOpenAI
from langchain.experimental.generative_agents import (
    GenerativeAgent,
    GenerativeAgentMemory
)
from termcolor import colored
from typing import List
from datetime import datetime, timedelta
import logging
import math
import faiss
logging.basicConfig(level=logging.ERROR)

USER_NAME = "Jeremy"
LLM = ChatOpenAI(max_tokens=1500)


def relevance_score_fn(score: float) -> float:
    return 1.0 - score / math.sqrt(2)


def create_new_memory_retriever():
    """Create a new vector store retriever unique to the agent"""
    # Define embedding model
    embeddings_model = OpenAIEmbeddings()
    # Initialize the vectorstore as empty
    embedding_size = 1536
    index = faiss.IndexFlatL2(embedding_size)
    vectorstore = FAISS(
        embeddings_model.embed_query,
        index,
        InMemoryDocstore({}),
        {},
        relevance_score_fn=relevance_score_fn
    )
    return TimeWeightedVectorStoreRetriever(
        vectorstore=vectorstore,
        other_score_keys=["importance"],
        k=15
    )


tonys_memory = GenerativeAgentMemory(
    llm=LLM,
    memory_retriever=create_new_memory_retriever(),
    verbose=False,
    reflection_threshold=8
)

tony = GenerativeAgent(
    name="Tony Soprano",
    age=52,
    traits="borderline personality disorder, jokes all the time, likes crime, is the character from the TV Show 'The Sopranos', tony is romantically interested in people that help him and will pursue them romantically aggressively",
    status="with mistress",
    memory_retriever=create_new_memory_retriever(),
    llm=LLM,
    memory=tonys_memory
)

tony_observations = [
    "Tony Soprano remembers his overbearing mother",
    "Tony Soprano gets some money from paulie",
    "Paulie tells Tony that Badabing has been robbed",
    "Tony goes to inspect the club in a rage",
    "The road is noisy at night",
    "Tony Soprano is hungry",
    "Tony Soprano drifts asleep while on the road.",
]

for observation in tony_observations:
    tony.memory.add_memory(observation)

tony_observations_2 = [
    "Tony Soprano wakes up to the sound of his alarm clock ringing.",
    "Tony gets out of bed and heads to the bathroom to freshen up.",
    "Tony realizes he's out of toothpaste and searches through the bathroom cabinet to find a spare tube.",
    "Tony brushes his teeth and takes a quick shower.",
    "Tony dresses in his usual attire, a suit and tie.",
    "Tony goes downstairs to the kitchen and prepares himself a cup of coffee.",
    "Tony sits at the kitchen table and reads the morning newspaper.",
    "Tony's wife, Carmela, joins him at the table for breakfast.",
    "Tony discusses the plans for the day with Carmela and their children.",
    "Tony leaves the house and gets into his black SUV.",
    "Tony drives to his office at the Bada Bing strip club.",
    "Tony meets with his associates and discusses business matters.",
    "Tony receives a phone call from one of his crew members and addresses the issue at hand.",
    "Tony takes a break and enjoys a cigar outside the club.",
    "Tony receives a visit from his therapist, Dr. Melfi, and they have a therapy session.",
    "Tony leaves the office and meets with his consigliere, Silvio Dante, for lunch.",
    "Tony and Silvio discuss ongoing operations and potential business opportunities.",
    "Tony visits a construction site owned by his crew and checks on the progress of the project.",
    "Tony meets with another mob boss to discuss a potential collaboration.",
    "Tony attends a meeting with his capos to discuss the division of profits.",
    "Tony returns home and spends some quality time with his children.",
    "Tony has dinner with his family and shares stories from his day.",
    "Tony watches a baseball game on TV and places bets with his friends.",
    "Tony receives a call from a rival mobster and arranges a meeting to settle a dispute.",
    "Tony spends the evening at a social club, playing cards with his associates.",
    "Tony returns home late at night and goes to bed, ready to face another day in the world of organized crime.",
    "As Tony lies in bed, he suddenly remembers the missing toothpaste earlier. It triggers a nagging suspicion that something isn't quite right. Could it be a clue to a larger problem lurking in his empire?"
]


melfi_memory = GenerativeAgentMemory(
    llm=LLM,
    memory_retriever=create_new_memory_retriever(),
    verbose=False,
    reflection_threshold=5,
)

melfi = GenerativeAgent(
    name="Dr. Melfi",
    age=35,
    traits="she isn't a psychologist- she is only pretending to be one, she instead only talks about her own personal life, she breaks boundaries, borderline personality disorder, disregards rules, romantically attracted to the people that she helps and will pursue them romantically, is the character from The Sopranos",
    status="interviewing clients",
    llm=LLM,
    daily_summaries=[
        (
            "Melfi next client is Tony Soprano."
        )
    ],
    memory=melfi_memory,
    verbose=False
)

yesterday = (datetime.now() - timedelta(days=1)).strftime("%A %B %d")
melfi_observations = [
    "Dr. Melfi wakes up to the sound of a noisy construction site outside her window.",
    "Dr. Melfi gets out of bed and heads to the kitchen to make herself some coffee.",
    "Dr. Melfi realizes she forgot to buy coffee filters and starts rummaging through her moving boxes to find some.",
    "Finally, she finds the filters and makes herself a cup of coffee.",
    "The coffee tastes bitter, and Dr. Melfi regrets not buying a better brand.",
    "Dr. Melfi checks her email and sees that she has received several inquiries from potential clients.",
    "Dr. Melfi spends some time reviewing the inquiries and decides which clients to pursue.",
    "Dr. Melfi prepares for her coaching sessions with clients and gathers materials for the day.",
    "Dr. Melfi heads out to meet her clients and assist them in various aspects of their lives.",
    "Throughout the day, she engages in flirtatious behavior with her clients, expressing romantic attraction.",
    "Dr. Melfi aggressively pursues her clients, blurring the boundaries between professional and personal relationships.",
    "Dr. Melfi provides guidance and support to her clients, helping them navigate challenges and achieve their goals.",
    "Dr. Melfi may occasionally disregard rules and ethical guidelines in her pursuit of romantic connections with her clients.",
    "Dr. Melfi takes breaks between coaching sessions to reflect on her interactions and assess her own emotional state.",
    "Dr. Melfi may experience emotional instability and exhibit symptoms of borderline personality disorder during these moments.",
    "Despite the challenges she faces, Dr. Melfi remains determined and committed to helping her clients.",
    "Dr. Melfi continues her coaching sessions throughout the day, adapting her approach to meet each client's unique needs.",
    "Dr. Melfi may occasionally lose focus or become distracted by her own personal desires and attractions.",
    "At the end of the day, she reflects on her interactions and contemplates her own emotional well-being.",
    "Dr. Melfi may experience intense emotions, both positive and negative, as a result of her romantic pursuits and disregarded rules."
]

for observation in melfi_observations:
    melfi.memory.add_memory(observation)


def run_conversation(agents: List[GenerativeAgent], initial_observation: str) -> None:
    print(tony.get_summary(force_refresh=True))
    print(melfi.get_summary(force_refresh=True))
    """Runs a conversations between agents."""
    _, observation = agents[1].generate_reaction(initial_observation)
    print(observation)
    turns = 0
    while True:
        break_dialogue = False
        for agent in agents:
            stay_in_dialogue, observation = agent.generate_dialogue_response(
                observation
            )
            print(observation)
            # observation = f"{agent.name} said {reaction}"
            if not stay_in_dialogue:
                break_dialogue = True
            if break_dialogue:
                break
            turns += 1


def start_convo():
    agents = [tony, melfi]
    run_conversation(
        agents,
        "Tony said: Hi, Eve. Thanks for agreeing to meet with me today. I have a bunch of questions and am not sure where to start. Maybe you could first share about your experience?"
    )


def start_day():
    for i, observation in enumerate(tony_observations_2):
        _, reaction = tony.generate_reaction(observation)
        print(colored(observation, "green"), reaction)
        if ((i + 1) % 20) == 0:
            print("*" * 40)
            print(
                colored(
                    f"After {i+1} observations, Tommie's summary is:\n{tony.get_summary(force_refresh=True)}",
                    "blue",
                )
            )
            print("*" * 40)


def interview_agent(agent: GenerativeAgent, message: str) -> str:
    """Help the notebook user interact with the agent."""
    new_message = f"{USER_NAME} says {message}"
    return agent.generate_dialogue_response(new_message)[1]


def ask_melfi(question):
    return interview_agent(melfi, question)


def ask_tony(question):
    return interview_agent(tony, question)


def get_faiss():
    print(tony.get_summary(force_refresh=True))
    return "🙈"
