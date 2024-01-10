from .get_openai_query import (
    llm_clean_information,
    llm_create_keywords,
    apply_regexes,
)

from .get_search_query import get_google_query, get_bing_query

import openai
import json

# from llama_cpp import Llama


class DavinciActionLayer:
    def __init__(self):
        with open("openai_conf.json", "r") as f:
            conf = json.load(f)
            self.api_key = conf["key"]
            self.model = "text-davinci-003"
            self.role = conf["role"]
        self.ask = self._create_service()
        self.max_length = 4096
        self.logfile = "gpt-reasoning-log.txt"

    def _create_service(self):
        openai.api_key = self.api_key
        # create service function that takes in a prompt and returns a response
        def service(prompt: str, stop: str):
            response = openai.Completion.create(
                model=self.model,
                max_tokens=512,
                stop=stop,
                prompt=prompt,
                temperature=0,
            )
            with open(self.logfile, "a") as f:
                f.write(
                    f"<task>\n<prompt>\n{prompt.strip()}\n</prompt>\n<response>\n{response['choices'][0]['text']}\n</response>\n</task>\n"
                )

            return response["choices"][0]["text"]

        return service


# class LLAMAActionLayer:
#     def __init__(self):
#         self.max_length = 512
#         self.ask = self._create_service()
#         self.model = Llama(
#             model_path=f"/home/blee/code/eclair_actions/model/llama-2-13b-chat.ggmlv3.q8_0.bin",
#             n_ctx=1024,
#             n_gpu_layers=1000,
#         )

#     def _create_service(self):
#         def service(prompt: str, stop: str):
#             res = self.model(prompt, max_tokens=512, stop=stop, echo=False)["choices"][
#                 0
#             ]["text"]
#             with open("llama-reasoning-log.txt", "a") as f:
#                 f.write(
#                     f"<task>\n<prompt>\n{prompt}\n</prompt>\n<response>\n{res}\n</response>\n</task>\n"
#                 )
#             return res

#         return service


# LAYER = LLAMAActionLayer()
LAYER = DavinciActionLayer()
SEARCH_ENGINE = "bing" # google, bing

# def make_info_concise(text):
#     prompt = """
#     Eliminate repeated information in the following text.

#     {text}
#     """.replace("{text}", text)

#     response = LAYER.ask(prompt, stop="\n")
#     return response


def last_line(text):
    if "\n" not in text:
        last_line = text
    else:
        last_line = text.split("\n")[-1]
    for i in range(1, len(text.split("\n"))):
        if text.split("\n")[-i] != "":
            last_line = text.split("\n")[-i]
            break
    return last_line


def gather_info(query):
    if SEARCH_ENGINE == "google":
        page_text = get_google_query(query)
    elif SEARCH_ENGINE == "bing":
        page_text = get_bing_query(query)
    else:
        raise Exception("Invalid search engine")
    # print("got page text")
    keywords = llm_create_keywords(query)["keywords"]
    # print("got keywords")
    information = apply_regexes(page_text, keywords, n=100)
    # print("got information")
    information = llm_clean_information("---".join(information), query)
    # print("got cleaned information")
    information = [str(info["relevant_information"]) for info in information]
    information = list(set(information))
    # return make_info_concise(" ".join(information))
    return " ".join(information)


def get_question(text):
    final_line = last_line(text)

    if "Follow up:" not in final_line:
        # print("ERROR: no follow up question found")
        return None
    if ":" not in final_line:
        after_colon = final_line
    else:
        after_colon = final_line.split(":")[-1]
    if " " == after_colon[0]:
        after_colon = after_colon[1:]
    if "?" != after_colon[-1]:
        # print("ERROR: no question mark found")
        return None
    return after_colon


def self_ask(initial_question, context=[]):
    # print(initial_question)
    prompt = [
        """Question: Who lived longer, Muhammad Ali or Alan Turing?
Are follow up questions needed here: Yes.
Follow up: How old was Muhammad Ali when he died?
Intermediate answer: Muhammad Ali was 74 years old when he died.
Follow up: How old was Alan Turing when he died?
Intermediate answer: Alan Turing was 41 years old when he died.
So the final answer is: Muhammad Ali 

Question: When was the founder of craigslist born?
Are follow up questions needed here: Yes.
Follow up: Who was the founder of craigslist?
Intermediate answer: Craigslist was founded by Craig Newmark.
Follow up: When was Craig Newmark born?
Intermediate answer: Craig Newmark was born on December 6, 1952.
So the final answer is: December 6, 1952

Question: Who was the maternal grandfather of George Washington?
Are follow up questions needed here: Yes.
Follow up: Who was the mother of George Washington?
Intermediate answer: The mother of George Washington was Mary Ball Washington.
Follow up: Who was the father of Mary Ball Washington?
Intermediate answer: The father of Mary Ball Washington was Joseph Ball.
So the final answer is: Joseph Ball 

Question: What is there to see near Sedona, Arizona?
Are follow up questions needed here: No.
Follow up: What is there to see near Sedona, Arizona?

Question: Are both the directors of Jaws and Casino Royale from the same country? 
Are follow up questions needed here: Yes. 
Follow up: Who is the director of Jaws? 
Intermediate Answer: The director of Jaws is Steven Spielberg. 
Follow up: Where is Steven Spielberg from? 
Intermediate Answer: The United States. 
Follow up: Who is the director of Casino Royale? 
Intermediate Answer: The director of Casino Royale is Martin Campbell. 
Follow up: Where is Martin Campbell from? 
Intermediate Answer: New Zealand. 
So the final answer is: No

Question: """,
        """
Are follow up questions needed here:""",
    ]

    prompt = prompt[0] + initial_question + prompt[1]
    response = LAYER.ask(prompt, stop="Intermediate answer:")
    while "Follow up:" in last_line(response):
        prompt += response
        question = get_question(response)
        if question is not None:
            # print(question)
            answer = gather_info(question)

        if answer is not None:
            prompt += "\nIntermediate answer: " + answer
            # print(answer)
            response = LAYER.ask(prompt, stop="Intermediate answer:")
        else:
            prompt += "\nIntermediate answer: "
            # print("ERROR: no answer found")
            gpt_answer = LAYER.ask(
                prompt, stop=["\n" + "Follow up:", "\nSo the final answer is:"]
            )
            prompt += gpt_answer

    if "So the final answer is:" not in response:
        prompt += "\nSo the final answer is: "
        # print("ERROR: no final answer found")
        response = LAYER.ask(prompt, stop="\n")
    return response.replace("So the final answer is: ", "").replace("No.\n", "")
