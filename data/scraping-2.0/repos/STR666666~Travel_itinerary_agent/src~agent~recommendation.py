import json
import re
import time
from langchain import LLMChain
from langchain.chat_models import ChatOpenAI
from src.agent.prompt_template import GeneralPromptTemplate
from src.tools.travel.google_map_client import google_map_client
from src.tools.travel.scenicspot import get_api_recommendation
from src.utils.debug import log_info, DEBUG

class Recommender:

    def __init__(self, city_name, days, user_preference):
        self.city_name = city_name
        self.days = days
        self.user_preference = user_preference
        with open("src/prompts/Travel/sight_recommendation.txt") as f:
            recommend_prompt = f.read()
        recommend_prompt_template = GeneralPromptTemplate(template=recommend_prompt,
                                                           input_variables=["city_name", "user_preference", "days","api_attractions"])
        chat_llm = ChatOpenAI(temperature=1, model_name="gpt-3.5-turbo-16k-0613")
        self.llm_chain = LLMChain(llm=chat_llm, prompt=recommend_prompt_template, verbose=DEBUG)
        self.gmaps = google_map_client()

    def recommend_sights(self):
        with open("src/prompts/Travel/cities.txt") as f:
            cities_prompt = f.read()
        recommend_prompt_template = GeneralPromptTemplate(template=cities_prompt,
                                                           input_variables=["cities"])
        chat_llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo")
        llm_chain = LLMChain(llm=chat_llm, prompt = recommend_prompt_template, verbose=DEBUG)
        api_recommendation = get_api_recommendation(self.city_name.strip())
        api_recommendation_text =""
        if api_recommendation["resp"]["RespCode"]=="200":
            for r_spot in api_recommendation["data"]["record"]:
                leave={}
                leave["sight_name"]=r_spot["spot"]
                leave["type"]=r_spot["type"]
                leave["recommend_duration"]=r_spot["visittime"]
                leave["open_time"]=r_spot["opentime"]
                api_recommendation_text = api_recommendation_text+str(leave)+"\n"
            
        recommended_sights = self.llm_chain(
            {"city_name": self.city_name, "user_preference": self.user_preference, "days": self.days, "api_attractions": api_recommendation_text},
            return_only_outputs=True)['text']
        
        regex = r"""[({"sight_name":.*, "recommend_reason":.*, "recommend_duration:.*"},)*{"sight_name":.*, "recommend_reason":.*, "recommend_duration:.*, ""recommend_play_time":.*"}]"""
        match = re.search(regex, recommended_sights)
        if match:
            li_recommendation = eval(recommended_sights)
            for k in li_recommendation:
                k["recommend_duration"]=int(k["recommend_duration"])
            return li_recommendation
        log_info(li_recommendation)

    def get_recommend_feedback(self, recommend_attractions):
        with open("src/prompts/Travel/attraction_recommend_feedback.txt") as f:
            feedback_prompt = f.read()
        feedback_prompt_template = GeneralPromptTemplate(template=feedback_prompt,
                                                        input_variables=["recommend_attractions", "user_preference","city_name","days"])
        chat_llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo-16k-0613")
        llm_chain = LLMChain(llm=chat_llm, prompt=feedback_prompt_template, verbose=DEBUG)
        feedback = llm_chain({"recommend_attractions": recommend_attractions, "city_name": self.city_name, "days": self.days,
                                "user_preference": self.user_preference}, return_only_outputs=True)['text']
        return feedback

    def edit_recommend_sights(self, recommend_attractions, feedback):
        with open("src/prompts/Travel/edit_recommend_sights.txt") as f:
            edit_prompt = f.read()
        edit_prompt_template = GeneralPromptTemplate(template=edit_prompt,
                                                        input_variables=["recommend_attractions", "user_preference","feedback","city_name","days"])
        chat_llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo-16k-0613")
        llm_chain = LLMChain(llm=chat_llm, prompt=edit_prompt_template, verbose=DEBUG)
        re_recommended_sights = llm_chain(
            {"recommend_attractions": recommend_attractions, "user_preference": self.user_preference, "feedback": feedback, "city_name": self.city_name, "days": self.days},
            return_only_outputs=True)['text']
        regex = r"""[({"sight_name":.*, "recommend_reason":.*, "recommend_duration:.*"},)*{"sight_name":.*, "recommend_reason":.*, "recommend_duration:.*"}]"""
        match = re.search(regex, re_recommended_sights)
        if match:
            return eval(re_recommended_sights)

    def search_famous_sights(self):
        recommended_sights = self.recommend_sights()
        log_info(json.dumps(recommended_sights, indent=4))
        recommend_feedback = self.get_recommend_feedback(recommended_sights)
        log_info(recommend_feedback)
        edited_recommended_sights = self.edit_recommend_sights(recommended_sights, recommend_feedback)
        log_info(json.dumps(edited_recommended_sights, indent=4))
        all_sights = []
        for sight in edited_recommended_sights:
            search_info = self.gmaps.find_place(sight["sight_name"] + f",{self.city_name}")
            if search_info is None or len(search_info) == 0:
                continue
            search_info["recommend_reason"] = sight["recommend_reason"]
            search_info["recommend_duration"] = sight["recommend_duration"]
            search_info["recommend_play_time"] = sight["recommend_play_time"]
            all_sights.append(search_info)
            time.sleep(0.3)

        log_info(json.dumps(all_sights, indent=4))
        return all_sights