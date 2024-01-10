import requests
import os
import openai
from langchain.llms import OpenAI

from datetime import datetime
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.memory import VectorStoreRetrieverMemory
from langchain.chains import ConversationChain
from langchain.prompts import PromptTemplate

import faiss

from langchain.docstore import InMemoryDocstore
from langchain.vectorstores import FAISS

KEY = "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"

class Janus:
    embedding_size = 1536 # Dimensions of the OpenAIEmbeddings
    index = faiss.IndexFlatL2(embedding_size)
    embedding_fn = OpenAIEmbeddings().embed_query

    def __init__(self, visions, goals, attributes, biography):
        self.visions = visions
        self.goals = goals
        self.attributes = attributes
        self.biography = biography 
        self.janus = OpenAI(openai_api_key=KEY)
        self.vectorstore = FAISS(embedding_fn, index, InMemoryDocstore({}), {})

    def set_entries(self, entries):
        self.entries = entries


    def progression(self):
        progression_prompt = """
        **Task one is progression.**
        
        Evaluate the progression in the journal entry on a scale from 1 to 100. 
        - A score of 1 indicates no progress towards goals/visions. In fact, as per the defined attributes, the user may have regressed.
        - A score of 100 indicates that the user has made significant progress or completed a goal/visions. 
        - A score of 50 percent means the user has attempted progress towards their goals/visions, but has not made progress.
        Scores between these three percentages should be determined based on attributes. If a user is making correct steps towards a goal,
        they have a higher score. 
        """
        return progression_prompt

    def suggestions(self): 
        suggestions_prompt = """
        **Task two is suggession.** 
        
        First provide two to three sentences that summarize whether the user is on track to meet their goals.
        Then, provide two to three suggestions for how they can improve my progress towards their goals.
        """
        return suggestions_prompt

    def get_past_entries(self, days=7):
        return entries[-days:]

    def summarize(self, weekly = False):
        if (weekly) :
            entries = get_past_entries()
            summary_prompt = f"""
            **Task three is summarization.**
            
            Summarize the following entries. 

            {entries}"""
        else :
            summary_prompt = ""

        return summary_prompt

    def expand(self, compressed):
        expanded_str = ""
        for i in compressed:
            expanded_str += "- " + i + "\n"
        return expanded_str

    def create_memory(self, k=3):
        # In actual usage, you would set `k` to be a higher value, but we use k=1 to show that
        # the vector lookup still returns the semantically relevant information
        retriever = self.vectorstore.as_retriever(search_kwargs=dict(k=1))
        return VectorStoreRetrieverMemory(retriever=retriever)



    def converse(self):
        memory = self.create_memory()


            

        # When added to an agent, the memory object can save pertinent information from conversations or used tools
        memory.save_context({"input": "My favorite food is pizza"}, {"output": "that's good to know"})
        memory.save_context({"input": "My favorite sport is soccer"}, {"output": "..."})
        memory.save_context({"input": "I don't the Celtics"}, {"output": "ok"}) #




    def converse(self, entry, weekly=False):
        progression, suggestions, summary = self.progression(), self.suggestions(), "placeholder\n"

        TEMPLATE = f"""
        You are Janus, an application that evaluates user journal entries. Your primary goal is to provide constructive feedback for users
        to optimize their daily life based on defined goals, visions, and attributes.

        Relevant Information:
        
        These are the user's visions, goals, and attributes. Attributes characterize how user's would like to achieve goals. User goals are stepping stones for
        visions, which are ultimate aspirations.

        Visions: 
        {self.expand(self.visions)}
        Goals: 
        {self.expand(self.goals)}
        Attributes: 
        {self.expand(self.attributes)}

        This is the user's biography:
        {self.biography}

        Here are the tasks you will perform:
        {progression}
        {suggestions}
        {summary}
        Return your answers in a json with "task":"answer". Your answers should speak to the user using "you", etc. The journal entries are below. 

        {entry}
        """
        
        # prediction = self.janus.predict(TEMPLATE)

        return TEMPLATE

visions = [
    "Build a family with two children that live meaningful lives",
    "Buy a vacation home on the beach and go there a month a year",
    "Become more spiritually connected with the earth"
]
goals = [
    "Finish learning to code and get a software engineering job",
    "Find another job besides my artist job",
    "Make enough money to move into a nicer apartment",
    "Quit partying and using MDMA"
]
attributes = [
    "Care more about learning",
    "Pay more attention to the cleanliness of my area",
    "Become more motivated in achieving my goals",
    "Spend less time on instant gratification"
]
biography = "She is a 35 year old woman who studied political science at her local community college. She is now a painter. She struggles, but she's able to make a livable income (40k a year). She parties a lot and often smokes weed and does psychedelics, which significantly affects her productivity. She recently had an awakening when meditating and she realized that she needs to turn her life around. its been hard for her though because she is stuck into so many of her old habits. Many days she gets closer to her goals, on other days she falls back into her old self.exit"

entry = """It's funny how the universe sends you signs. During my meditation, it hit me - I've spent years as a leaf in the wind, blowing wherever life took me. There's beauty in that, sure, but there's a void too, a sense of something lacking. I've spent so much time enjoying the moment without truly considering the future. The partying, the weed, the mdma, all the moments that felt freeing but chained me in the long run.
Today, I opened up my computer and started a coding lesson. I'd been putting it off forever. I got through an hour, which isn't a lot, but it's a start. My hands felt more at home holding a paintbrush, but the thought of securing a stable future and that dream beach house kept me going. I imagined the sound of waves crashing, my future kids playing in the sand... I want that life."""

janus_1 = Janus(visions, goals, attributes, biography)
print(janus_1.converse(entry, weekly=False))