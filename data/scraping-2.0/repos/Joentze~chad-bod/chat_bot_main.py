"""Running LLM Scripts"""
import json
from typing import List
from datetime import datetime
from dataclasses import dataclass
import openai
from secret_keys import TELEGRAM_API_KEY, OPEN_AI_KEY
from build_supabase import get_context_from_supabase
from prompts import get_prompt, insert_context_to_prompt
from telegram_helper import edit_message
from redis_handler import insert_message_history, get_message_history
from llm_functions.function_map import function_map, llm_functions

MODEL_NAME = "gpt-3.5-turbo-0613"
MAX_NUM_TOKEN_TELEGRAM = 50
EXCEED_TOKEN_MESSAGE = """```You've exceeded the token limit for this message, please rephrase into a shorter statement...```"""
openai.api_key = OPEN_AI_KEY

# davinci = OpenAI(model_name=MODEL_NAME,
#                  openai_api_key=OPEN_AI_KEY, temperature=0, max_tokens=1000)


@dataclass
class TelegramQuery:
    chat_id: str
    message_id: str
    query: str


def is_within_token_limit(message: str) -> bool:
    """checks if message sent is within character limit"""
    return len(message)//4 <= MAX_NUM_TOKEN_TELEGRAM


# def run_llm(question: str):
#     """runs open ai llm"""
#     contexts = get_context_from_supabase(question, 0.8, 3)
#     llm_chain = LLMChain(prompt=get_prompt(contexts), llm=davinci)
#     response = llm_chain.run(question)
#     return response


def respond_with_llm(configs):
    """edits specific telegram bot message"""
    query = TelegramQuery(
        chat_id=configs["chat_id"], message_id=configs["message_id"], query=configs["query"])
    message_history = get_message_history(query.chat_id)
    response = chat_completion(query.chat_id, query.query, message_history)
    edit_message(API_KEY=TELEGRAM_API_KEY, message_id=query.message_id,
                 chat_id=query.chat_id, new_message=response)


def chat_completion(chat_id: str, curr_query: str, history: List[object]) -> str:
    """sends query to LLM"""
    contexts = get_context_from_supabase(curr_query, 0.8, 3)
    prompt = insert_context_to_prompt(curr_query, contexts)

    message_history = [
        {"role": "system", "content": "You are Chad Bod, a Singapore Management University Student Helper. You do not help students with any of their school work, you can only advise them briefly"},
        *history,
        {"role": "user", "content": prompt}
    ]
    print(message_history)
    completion = openai.ChatCompletion.create(
        model=MODEL_NAME,
        temperature=0,
        messages=message_history,
        functions=llm_functions

    )

    # message = completion['choices'][0]['message']['content']
    response_body = completion['choices'][0]['message']
    if "function_call" in response_body:
        func = response_body["function_call"]
        function_name = func["name"]
        args = func["arguments"]
        args = json.loads(args)
        message = function_map[function_name](**args)
    else:
        message = response_body["content"]
    insert_message_history(chat_id=chat_id, message={
                           "role": "assistant", "content": message})
    return message


if __name__ == "__main__":
    t1 = datetime.now()
    # print(run_llm("how many libraries are there in smu"))
    messages = get_message_history("549991017")
    # print(type(messages))
    test = [{'role': 'system', 'content': 'You are Chad Bod, a Singapore Management University Student Helper.'}, {'role': 'user', 'content': 'who is the president of smu'}, {'role': 'user', 'content': 'who is kyriakos'}, {'role': 'user', 'content': 'how do i bid for classes?'}, {'role': 'user', 'content': 'how do i start planning for exchange'}, {'role': 'user',
                                                                                                                                                                                                                                                                                                                                                              'content': "\nRoleplay as the following:\nYou are an enthusiastic student helper of Singapore Management University. You respond to student's questions based on the context in a direct manner. If you do not know how to respond to the question, just say you do not know, do not come up with your own answers. quote the sources from context.\n\ncontext:\nWhat should I do if I do not have sufficient e$? Additional e$ will not be allocated as all students are given the same amount of e$ and e-pt in each term throughout the years of study in SMU. Please adjust your e$ bids accordingly so that you can bid for additional courses.But if you do not have sufficient e$ to bid for courses in your final term, please proceed to bid for the required courses with all your e$ until the end of Round 1B. You might be able to get your bids. If you are still unable to have successful bids, please consult your school manager for advice. (source: https://oasis.smu.edu.sg/Pages/faq.aspx)\nHow can I check for the applicable course area(s) for a course? Navigate toBOSS> BOSS Bidding > Plan & Bid > Add to Cart > Course Search to search for courses under a specific course area.You should check the course area of the class you wish to enrol in, as the course area(s) may change over time. (source: https://oasis.smu.edu.sg/Pages/faq.aspx)\n\nquestion:\nhow should i plan for bidding\n\nanswer:\n"}]
    completion = chat_completion(
        "549991017", "what is lks", messages)

    print(completion)
    # respond_with_llm({
    #     "chat_id": 549991017,
    #     "message_id": 73,
    #     "query": "what are the requirements for dean's list"
    # })
    print("total time taken: ", datetime.now()-t1)
