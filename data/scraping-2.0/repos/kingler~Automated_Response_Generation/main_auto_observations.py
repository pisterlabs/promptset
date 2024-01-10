import re
from datetime import datetime, timedelta
from typing import List, Optional, Tuple
from termcolor import colored

from pydantic import BaseModel, Field

from langchain import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.docstore import InMemoryDocstore
from langchain.embeddings import OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.retrievers import TimeWeightedVectorStoreRetriever
from langchain.schema import BaseLanguageModel, Document
from langchain.vectorstores import FAISS

import math
import faiss

from gen_agent import GenerativeAgent

USER_NAME = "echo" # The name you want to use when interviewing the agent.
LLM = ChatOpenAI(max_tokens=1500) # Can be any LLM you want.

import math
import faiss

import openai
from typing import List
import re
import time

print(colored("WELCOME TO THE VIRTUAL WORLD OF TOMMIE AND EVE POWERED BY GPT AND LANGCHAIN", "green", attrs=["bold"]))
print("\n")

def relevance_score_fn(score: float) -> float:
    """Return a similarity score on a scale [0, 1]."""
    # This will differ depending on a few things:
    # - the distance / similarity metric used by the VectorStore
    # - the scale of your embeddings (OpenAI's are unit norm. Many others are not!)
    # This function converts the euclidean norm of normalized embeddings
    # (0 is most similar, sqrt(2) most dissimilar)
    # to a similarity function (0 to 1)
    return 1.0 - score / math.sqrt(2)

def create_new_memory_retriever():
    """Create a new vector store retriever unique to the agent."""
    # Define your embedding model
    embeddings_model = OpenAIEmbeddings()
    # Initialize the vectorstore as empty
    embedding_size = 1536
    index = faiss.IndexFlatL2(embedding_size)
    vectorstore = FAISS(embeddings_model.embed_query, index, InMemoryDocstore({}), {}, relevance_score_fn=relevance_score_fn)
    return TimeWeightedVectorStoreRetriever(vectorstore=vectorstore, other_score_keys=["importance"], k=15)    

tommie = GenerativeAgent(name="Tommie", 
              age=25,
              traits="anxious, likes design", # You can add more persistent traits here 
              status="looking for a job", # When connected to a virtual world, we can have the characters update their status
              memory_retriever=create_new_memory_retriever(),
              llm=LLM,
              daily_summaries = [
                   "Drove across state to move to a new town but doesn't have a job yet."
               ],
               reflection_threshold = 8, # we will give this a relatively low number to show how reflection works
             )

# print(tommie.get_summary()) # currently there are no memories

# # Let's add a memory
# tommie.add_memory("Tommie moved to a new town and is looking for a job.")


# print(tommie.get_summary(force_refresh=True)) # now we have a memory

# #we can add more memories
tommie_memories = [
    "Tommie is looking for a job.",
    "Tommie exercises every day.",
    "Tommie's favorite food is pizza.",
    "Tommie likes to play video games with friends.",
    "Tommie is a good friend.",

]

for memory in tommie_memories:
    tommie.add_memory(memory)

# print(tommie.get_summary(force_refresh=True)) # now we have more memories

def interview_agent(agent: GenerativeAgent, message: str) -> str:
    """Help the notebook user interact with the agent."""
    new_message = f"{USER_NAME} says {message}"
    return agent.generate_dialogue_response(new_message)[1]


eve = GenerativeAgent(name="Eve", 
              age=34, 
              traits="curious, helpful", # You can add more persistent traits here 
              status="N/A", # When connected to a virtual world, we can have the characters update their status
              memory_retriever=create_new_memory_retriever(),
              llm=LLM,
              daily_summaries = [
                  ("Eve started her new job as a career counselor last week and received her first assignment, a client named Tommie.")
              ],
                reflection_threshold = 5,
             )

yesterday = (datetime.now() - timedelta(days=1)).strftime("%A %B %d")
eve_memories = [
    "Eve overhears her colleague say something about a new client being hard to work with",
    "Eve wakes up and hear's the alarm",
    "Eve eats a boal of porridge",
    "Eve helps a coworker on a task",
    "Eve plays tennis with her friend Xu before going to work",
    "Eve overhears her colleague say something about Tommie being hard to work with",
    
]
for memory in eve_memories:
    eve.add_memory(memory)

def run_conversation(agents: List[GenerativeAgent], initial_observation: str) -> None:
    """Runs a conversation between agents."""
    _, observation = agents[1].generate_reaction(initial_observation)
    print(observation)
    turns = 0
    while True:
        break_dialogue = False
        for agent in agents:
            stay_in_dialogue, observation = agent.generate_dialogue_response(observation)
            print(observation)
            # observation = f"{agent.name} said {reaction}"
            if not stay_in_dialogue:
                break_dialogue = True   
        if break_dialogue:
            break
        turns += 1

agents = [tommie, eve]


def get_daily_events(character_name: str) -> List[str]:
    number_of_events = 4
    if character_name == "Tommie":
        character_name_summary = tommie
    elif character_name == "Eve":
        character_name_summary = eve
    messages = [
            {"role": "system", "content": f"You generate a python list of daily events for a character. Return exactly {number_of_events} events in the list."},
            {"role": "user", "content": f"""generate a list of exactly {number_of_events} daily events sequentially for the whole day for {character_name} with history: {character_name_summary.get_summary(force_refresh=True)}
             Sample output: [
    "character wakes up to the sound of a noisy construction site outside his window.",
    "character gets out of bed and heads to the kitchen to make himself some coffee.",
    "character realizes he forgot to buy coffee filters and starts rummaging through his moving boxes to find some.",
    "character finally finds the filters and makes himself a cup of coffee.",
    "The coffee tastes bitter, and Tommie regrets not buying a better brand.",
    "character checks his email and sees that he has no job offers yet.",
    "character spends some time updating his resume and cover letter.",
    "character heads out to explore the city and look for job openings.",
    "character sees a sign for a job fair and decides to attend.",
    "The line to get in is long, and Tommie has to wait for an hour.",
    "character meets several potential employers at the job fair but doesn't receive any offers.",
    "character leaves the job fair feeling disappointed.",
    "character stops by a local diner to grab some lunch.",
    "The service is slow, and Tommie has to wait for 30 minutes to get his food.",
    "character overhears a conversation at the next table about a job opening.",
    "character asks the diners about the job opening and gets some information about the company.",
    "character decides to apply for the job and sends his resume and cover letter.",
    "character continues his search for job openings and drops off his resume at several local businesses.",
    "character takes a break from his job search to go for a walk in a nearby park.",
    "A dog approaches and licks Tommie's feet, and he pets it for a few minutes.",
    "character sees a group of people playing frisbee and decides to join in.",
    "character has fun playing frisbee but gets hit in the face with the frisbee and hurts his nose.",
    "character goes back to his apartment to rest for a bit.",
    "A raccoon tore open the trash bag outside his apartment, and the garbage is all over the floor.",
    "character starts to feel frustrated with his job search.",
    "character calls his best friend to vent about his struggles.",
    "character's friend offers some words of encouragement and tells him to keep trying.",
    "character feels slightly better after talking to his friend.",
]
    You won't repeat the sample events in the output.
    change the "character" in the sample output to the name of the character you are generating events for.
    Don't include anything else in your response except only the python list of events between the brackets.
    response should include exactly {number_of_events} events only.
    """
    },
        ]
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages= messages,
    )
    # print("call to openai: ", messages)
    daily_events_text = response['choices'][0]['message']['content']
    # content of the response is a string, so we need to convert it to a list usin eval from
    daily_events = eval(daily_events_text)
    # print(daily_events)

    
    #print daily events for the character one by one colored blue
    print(colored(f"Daily events for {character_name}:", "green"))
    # print("\n")

    for i, event in enumerate(daily_events):
        print(colored(f"event {i+1} for {character_name}:", "blue"), event)
        time.sleep(0.1)
    print("\n")
    print("*" * 40)
        
    return daily_events
    


# Get daily events for both characters
tommie_daily_events = get_daily_events("Tommie")
# print(tommie_daily_events)
eve_daily_events = get_daily_events("Eve")
# print(eve_daily_events)

# check the length of the daily events list for both characters and pop the 2nd event until the longer one equals the shorter one
while len(tommie_daily_events) != len(eve_daily_events):
    print("number of daily events for Tommie and Eve are not equal. will pop the 2nd event from the longer list until they are equal")
    if len(tommie_daily_events) > len(eve_daily_events):
        tommie_daily_events.pop(1)
    else:
        eve_daily_events.pop(1)

# ... (rest of the code for creating the characters, Tommie and Eve) ...

# Modify the main loop to include user input every 5 observations
observation_count = 0
day = 1
while True:
    print(colored("START OF DAY ", "green"), day)
    print("\n")
    for i, (tommie_observation, eve_observation) in enumerate(zip(tommie_daily_events, eve_daily_events)):
        _, tommie_reaction = tommie.generate_reaction(tommie_observation)
        print(colored(tommie_observation, "green"), tommie_reaction)

        _, eve_reaction = eve.generate_reaction(eve_observation)
        print(colored(eve_observation, "green"), eve_reaction)

        observation_count += 1

        # checkpoint to interact with the agents
        if observation_count % 50 == 0:
            while True:
                user_action = input(colored("Do you want to [c]ontinue, get [s]ummary, [chat], [int]erview or [q]uit? ", "yellow")).lower()

                if user_action == 's':
                    print(f"Tommie's summary:\n{tommie.get_summary(force_refresh=True)}")
                    print(f"Eve's summary:\n{eve.get_summary(force_refresh=True)}")
                elif user_action == 'chat':
                    agents = [tommie, eve]
                    convo_starter = f"Tommie said: Hi Eve: ", input(colored("What should the first character say to start the conversation? ", "yellow"))
                    run_conversation(agents, convo_starter)
                elif user_action == 'c':
                    break
                elif user_action == 'int':
                    while True:
                        character_to_interview = input(colored("Which character do you want to interview? [t]ommie or [e]ve or [q]uit? ", "yellow")).lower()
                        if character_to_interview == 'q':
                            break
                        if character_to_interview not in ['t', 'e', 'q']:
                            print("Invalid character")
                            continue
                        if character_to_interview == 't':
                            interview_question = input(colored("What question do you want to ask to Tommie? [q] to quit ", "yellow"))
                            if interview_question == 'q':
                                break
                            print(f"Tommie's answer: {interview_agent(tommie, interview_question)}")
                        elif character_to_interview == 'e':
                            interview_question = input(colored("What question do you want to ask to Eve? [q] to quit ", "yellow"))
                            if interview_question == 'q':
                                break
                            print(f"Eve's answer: {interview_agent(eve, interview_question)}")
                elif user_action == 'q':
                    break

            if user_action == 'q':
                break




    print(colored("END OF DAY ", "red"), day)
    print("\n")
    print("-" * 40)
    print(colored("START OF DAY ", "green"), day + 1)

    day += 1
    # generate and append nest day's events
    new_daily_events_for_tommie = get_daily_events("Tommie")
    for event in new_daily_events_for_tommie:
        tommie_daily_events.append(event)
    new_daily_events_for_eve = get_daily_events("Eve")
    for event in new_daily_events_for_eve:
        eve_daily_events.append(event)


                


