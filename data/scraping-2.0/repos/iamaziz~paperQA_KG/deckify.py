from typing import List, Tuple
from functools import lru_cache

from gpt import OpenAIService
import streamlit as st

if "model" not in st.session_state:
    st.session_state.model = None
if "selected_model" not in st.session_state:
    st.session_state.selected_model = None



class BuildSlides:
    def __init__(self, docs: List[str], urls: List[str], paper_topic: str): #, num_slides: int = 10):
         self.docs = docs
         self.urls = urls
         self.topic = paper_topic
         llm = st.session_state.selected_model
         self.model = OpenAIService(model_name=llm)
         self.model_name = self.model.model_name
         st.session_state.model = self.model
        

    @staticmethod
    @st.cache_resource
    def deckify(prompt, openai = st.session_state.model):
        # measure time to run
        import time
        start = time.time()
        # openai = st.session_state.model # OpenAIService()
        res = openai.prompt(prompt)
        end = time.time()
        return res, end-start
    
    @staticmethod
    @st.cache_resource
    def chatify(user_prompt, system_prompt=None, assistant_prompt=None, openai = st.session_state.model):
        # measure time to run
        import time
        start = time.time()
        # openai = st.session_state.model # OpenAIService()
        res = openai.prompt(user_prompt=user_prompt, system_prompt=system_prompt)#, assistant_prompt=assistant_prom)
        end = time.time()
        return res, end-start
    
    @property
    def system_prompt(self) -> str:
        prompt = f"""All your answers should be in the context of "{self.topic}" based on the information in following paper abstracts:"""
        for i, doc, url in zip(range(1, len(self.docs)+1), self.docs, self.urls):
            if doc is not None:
                prompt += f"\n<START-OF-PAPER {i}>\n"
                prompt += f"\n{doc}\n"
                prompt += f"\n<PAPER-REFERENCE {url}>\n"
                prompt += f"\n<END-OF-PAPER {i}>\n"
        
        rules = f"""
        You should cite the paper number in your answers. And, add a link to the paper's url in the reference section.
        You should consider describing your answers mathematically using latex.
        You should consider adding a figure to your answers.
        You should consider adding a table to your answers.
        You should consider adding a citation to your answers.
        """
        prompt += rules

        return prompt
        


    @staticmethod
    @st.cache_data
    def build_prompt(docs: Tuple[str], urls: List[str], paper_topic: str, num_slides: int = 10):
            prompt = f""""""
            for i, doc, url in zip(range(1, len(docs)+1), docs, urls):
                if doc is not None:
                    prompt += f"\n<START-OF-PAPER {i}>\n"
                    prompt += f"\n{doc}\n"
                    prompt += f"\n<PAPER-REFERENCE {url}>\n" # NOTE: takes too long to run and cause weird behavior for citation formatting
                    prompt += f"\n<END-OF-PAPER {i}>\n"
            
            # NOTE: takes too long to run (TOC and reference slides are not necessary)
            prompt_with_reference_links = f"""
You are a researcher in the field of {paper_topic}.
You are reading the following paper abstracts about {paper_topic}.
Your goal is to build slides for a presentation about {paper_topic} based on the abstracts.
You should create at least {num_slides} slides that cover the main takeaways from all papers.
Each slide should have a title and a subtitle.
The slides should have introductory, table of contents, concluding, and reference slides.
The introductory slide must include a brief paragraph about the topic.
The concluding slide must include a brief paragraph to summarize the main takeaways.
The reference slide must include a list urls to the papers as hyperlinks.
The key takeaways should be in the form of bullet points.
Each bullet point must be a complete idea or sentence on its own.
You should add reference to the abstract number in each bullet point.
The slides should be formatted in markdown. 
The title and subtitle should be in header2 and header3 respectively.
Finally, here are the paper abstracts:

            {prompt}
            """
            prompt_ = f"""
You are a researcher in the field of {paper_topic}.
You are reading the following paper abstracts about {paper_topic}.
Your goal is to build slides for a presentation about {paper_topic} based on the abstracts.
You should create at least {num_slides} slides that cover the main takeaways from all papers.
Each slide should have a title and a subtitle.
The slides should have introductory, concluding, and reference slides.
The introductory slide must include a brief paragraph about the topic.
The concluding slide must include a brief paragraph to summarize the main takeaways.
The reference slide should have a list of urls to the papers as hyperlinks.
The key takeaways should be in the form of bullet points.
Each bullet point must be a complete idea or sentence on its own.
You should add reference to the paper number in each bullet point.
The slides should be formatted in markdown.
The title and subtitle should be in header2 and header3 respectively.
Finally, here are the paper abstracts:

            {prompt}
            """
            prompt = f"""
You are a researcher in the field of {paper_topic}.
You are reading the following paper abstracts about {paper_topic}.
Your goal is to build slides for a presentation about {paper_topic} based on the abstracts.
You should create at least {num_slides} slides that cover the main takeaways from all papers.
Each slide should have a title and a subtitle.
The slides should have introductory, concluding, and reference slides.
The introductory slide must include a brief paragraph about the topic.
The concluding slide must include a brief paragraph to summarize the main takeaways.
Each key idea should be in its own slide.
The reference slide should have a list of urls to the papers as hyperlinks.
The key takeaways should be in the form of bullet points.
Each bullet point must be a complete idea or sentence on its own.
You should add reference to the paper number in each bullet point.
The slides should be formatted in markdown.
The title and subtitle should be in header2 and header3 respectively.
Finally, here are the paper abstracts:

            {prompt}
            """
#             system_prompt = f"""
# You are a researcher in the field of {paper_topic}.
# You are reading the provided paper abstracts about {paper_topic}.
# Your goal is to build slides for a presentation about {paper_topic} based on the provided abstracts.
# From your understanding of the abstracts;
# You should create at least {num_slides} slides that cover the main takeaways from all papers.
# Each slide should have a title and a subtitle.
# The slides should have introductory and concluding slides.
# The introductory slide must include a brief paragraph about the topic.
# The concluding slide must include a brief paragraph to summarize the main takeaways.
# The key takeaways should be in the form of bullet points.
# Each bullet point must be a complete idea or sentence on its own.
# You should add reference to the abstract number where appropriate.
# The slides should be formatted in markdown. 
# The title and subtitle should be in header2 and header3 respectively.
#             """
            # user_prompt = f"""Can you create the slides as described above?"""
            # return user_prompt, system_prompt, assistant_prompt
            return prompt