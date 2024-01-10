import os
import openai
import torch
import backoff

from typing import List
from nltk.tokenize import sent_tokenize
from sentence_transformers import SentenceTransformer, util

from prompts import (
    action_decision_prompt,
    action_input_search_prompt,
    action_input_recommend_prompt,
    response_generation_with_knowledge,
    response_generation_without_knowledge_prompt,
    system_instruction,
    system_instruction_with_knowledge_prompt,
    summarize_chat_prompt
)
from recommend import RecommendModule
from search import SearchModule


class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


@backoff.on_exception(backoff.expo, openai.error.RateLimitError)
def completions_with_backoff(**kwargs):
    return openai.Completion.create(**kwargs)


@backoff.on_exception(backoff.expo, openai.error.RateLimitError)
def chat_completions_with_backoff(**kwargs):
    return openai.ChatCompletion.create(**kwargs)


def posprocess_result(generated_response: str) -> str:
    NEXT_SPEAKER = "\nShopper:"
    if NEXT_SPEAKER in generated_response:
        start = generated_response.find(NEXT_SPEAKER)
        return generated_response[:start]
    return generated_response


def closest_match(candidates: List[str], query: str, embedder):
    candidate_embeddings = embedder.encode(candidates, convert_to_tensor=True)
    query_embedding = embedder.encode(query, convert_to_tensor=True)
    cos_scores = util.cos_sim(query_embedding, candidate_embeddings)[0]
    top_results = torch.topk(cos_scores, k=1)
    score, idx = top_results
    return idx


def get_recommended_items_from_sentence(candidates, sentence, embedder, sim_threshold):
    query_embedding = embedder.encode(sentence.lower(), convert_to_tensor=True)
    final_rec_items = []
    for item in candidates:
        candidate = item.metadata["title"].lower()
        if candidate in sentence:
            final_rec_items.append((1.0, item))
            continue
        candidate_embedding = embedder.encode(candidate, convert_to_tensor=True)
        cos_scores = util.cos_sim(query_embedding, candidate_embedding)[0]
        top_results = torch.topk(cos_scores, k=1)
        sim_score = top_results.values.item()
        print(f"{sim_score}: {candidate}")
        if sim_score > sim_threshold:
            print(f"---> Recommending {candidate}")
            final_rec_items.append((sim_score, item))
    recommended_items = [item for sim_score, item in sorted(final_rec_items)]
    return recommended_items


def get_recommended_items(candidates, response, embedder, sim_threshold):
    # Filter out products that were not mentioned by the salesperson response
    sentences = sent_tokenize(response.lower())
    recommended_items = []
    recommended_titles = set()
    for sentence in sentences:
        mentioned_items = get_recommended_items_from_sentence(candidates, sentence, embedder, sim_threshold)
        for item in mentioned_items:
            if item.metadata["title"] in recommended_titles:
                continue
            else:
                recommended_items.append(item)
                recommended_titles.add(item.metadata["title"])
    return recommended_items


class SalesBot(object):

    def __init__(self):
        openai.organization = os.environ.get("OPENAI_API_ORG")
        openai.api_key = os.environ.get("OPENAI_API_KEY")

        self.search_module = SearchModule()
        self.recommend_module = RecommendModule()
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
        self.sim_threshold = 0.70
        self.total_tokens = 0
        self.openai_responses = []
        # gpt-3.5-turbo:    $0.002 / 1K total tokens
        # text-davinci-003: $0.02 / 1K total tokens
        self.gpt3_tokens = 0
        self.chatgpt_tokens = 0

    def openai_generate(self, prompt: str) -> str:
        response = completions_with_backoff(
            model="text-davinci-003",
            prompt=prompt,
            max_tokens=64,
            temperature=0.0,
        )
        self.gpt3_tokens += response['usage']['total_tokens']
        self.total_tokens += response['usage']['total_tokens']
        self.openai_responses.append(response)
        return response["choices"][0]["text"]

    def openai_chat_generate(self, prompt: str, instruction: str = 'Follow user instructions carefully') -> str:
        response = chat_completions_with_backoff(
            model="gpt-3.5-turbo-16k",
            messages=[
                {"role": "system", "content": instruction},
                {"role": "user", "content": prompt}
            ],
            max_tokens=128,
            temperature=0.0,
        )
        self.chatgpt_tokens += response['usage']['total_tokens']
        self.total_tokens += response['usage']['total_tokens']
        self.openai_responses.append(response)
        if response["choices"][0]["finish_reason"] != "stop":
            answer = response["choices"][0]["message"]["content"]
            print(f"{bcolors.FAIL}Rewrite: {answer}{bcolors.ENDC}")
            return self.openai_chat_generate(prompt=f"Rewrite the salesperson's response to be more consise and engaging:\nSalesperson: {answer}\nSalesperson:", instruction=instruction)
        return response["choices"][0]["message"]["content"]

    def openai_chat(self, prompt_messages) -> str:
        response = chat_completions_with_backoff(
            model="gpt-3.5-turbo-16k",
            messages=prompt_messages,
            max_tokens=256,
            temperature=0.0,
        )
        self.chatgpt_tokens += response['usage']['total_tokens']
        self.total_tokens += response['usage']['total_tokens']
        self.openai_responses.append(response)
        if response["choices"][0]["finish_reason"] == "stop":
            return response["choices"][0]["message"]["content"]
        elif response["choices"][0]["finish_reason"] == "length":
            if response["usage"]["prompt_tokens"] >= 4090:
                print(f"Input is too long, summarizing the chat...")
                print(prompt_messages)
                return self.summarize_and_reply(prompt_messages)
            else:
                answer = response["choices"][0]["message"]["content"]
                print(f"{bcolors.FAIL}Rewrite: {answer}{bcolors.ENDC}")
                return self.openai_chat([{
                    "role": "user",
                    "content": f"Rewrite the salesperson's response to be more concise and engaging:\nnSalesperson: {answer}\nnSalesperson:"
                }])
        else:
            return "Sorry, I cannot fulfill your request at this time. Please try again later."

    def compute_cost(self):
        return self.gpt3_tokens / 1000.0 * 0.02 + self.chatgpt_tokens / 1000.0 * 0.002

    def parse_chat_history(self, chat_history):
        messages = []
        for u in chat_history:
            if u.startswith("Shopper: "):
                text = u.replace("Shopper: ", "")
                messages.append({"role": "user", "content": text})
            elif u.startswith("Salesperson: "):
                text = u.replace("Salesperson: ", "")
                messages.append({"role": "assistant", "content": text})
        return messages

    def generate_knowledge_reply(self, input_txt, chat_history, knowledge):
        instruction = system_instruction_with_knowledge_prompt.format(knowledge=knowledge)
        prev_messages = self.parse_chat_history(chat_history)
        messages = [{"role": "system", "content": instruction}] + prev_messages + [{"role": "user", "content": input_txt}]
        return self.openai_chat(messages)

    def generate_non_knowledge_reply(self, input_txt, chat_history):
        prev_messages = self.parse_chat_history(chat_history)
        messages = [{"role": "system", "content": system_instruction}] + prev_messages + [{"role": "user", "content": input_txt}]
        return self.openai_chat(messages)

    def summarize_and_reply(self, prompt_messages):
        system_text = prompt_messages[0]["content"]
        input_txt = prompt_messages[-1]["content"]

        chat_history = ''
        for m in prompt_messages[1:-1]:
            if m['role'] == 'user':
                chat_history += f"\nUser: {m['content']}"
            elif m['role'] == 'assistant':
                chat_history += f"\nAssistant: {m['content']}"
        summarize_prompt = summarize_chat_prompt.format(chat_history=chat_history)
        chat_summary = self.openai_chat_generate(summarize_prompt).strip()
        system_text += f"\n\nChat summary:\n{chat_summary}"
        messages = [
            {"role": "system", "content": system_text},
            {"role": "user", "content": input_txt}
        ]
        try:
            final_response = self.openai_chat(messages)
            return final_response
        except Exception as e:
            print(e)
            return "Sorry, I cannot fulfill your request at this time. Please try again later."

    def generate(self, input_txt, chat_history):
        context = '\n'.join(chat_history)
        full_chat_history=f"{context}\nShopper: {input_txt}"
        action_prompt = action_decision_prompt.format(chat_history=context,input=input_txt)
        action_result = self.openai_generate(action_prompt).strip()
        chosen_action = 'None'
        recommended_items = []
        for action_key in ['Knowledge', 'Recommend', 'LookUpProductInfo']:
            if action_key in action_result:
                chosen_action = action_key
                break
        if chosen_action == 'Knowledge':
            prompt = action_input_search_prompt.format(chat_history=context, input=input_txt)
            query = self.openai_generate(prompt).strip()
            top_docs = self.search_module.top_docs(query=query, k=2)
            knowledge = "\n---\n".join([doc.page_content for doc in top_docs])
            print(f"{bcolors.OKGREEN}Action: {chosen_action}{bcolors.ENDC}")
            print(f"{bcolors.OKBLUE}Query: {query}{bcolors.ENDC}")
            print(f"{bcolors.OKBLUE}Knowledge: {knowledge}{bcolors.ENDC}")
            knwldge_prompt = response_generation_with_knowledge.format(chat_history=full_chat_history, action=chosen_action, query=query, knowledge=knowledge)
            result = self.openai_chat_generate(knwldge_prompt).strip()
        elif chosen_action == 'Recommend':
            prompt = action_input_recommend_prompt.format(chat_history=context, input=input_txt)
            query = self.openai_generate(prompt).strip()
            recommender_candidates = self.recommend_module.top_docs(query, k=4)
            # run through recommendation_template to narrow down options
            # pass recommender candidates as knowledge
            knowledge = "\n---\n".join([item.page_content for item in recommender_candidates])
            print(f"{bcolors.OKGREEN}Action: {chosen_action}{bcolors.ENDC}")
            print(f"{bcolors.OKBLUE}Query: {query}{bcolors.ENDC}")
            print(f"{bcolors.OKBLUE}Knowledge: {knowledge}{bcolors.ENDC}")

            knwldge_prompt = response_generation_with_knowledge.format(chat_history=full_chat_history, action=chosen_action, query=query, knowledge=knowledge)
            result = self.openai_chat_generate(knwldge_prompt).strip()
            recommended_items = get_recommended_items(recommender_candidates, result, self.embedder, self.sim_threshold)
        elif chosen_action == 'LookUpProductInfo':
            prompt = action_input_recommend_prompt.format(chat_history=context, input=input_txt)
            query = self.openai_generate(prompt).strip()
            recommender_candidates = self.recommend_module.top_docs(query, k=4)
            knowledge = "\n---\n".join([item.page_content for item in recommender_candidates])

            print(f"{bcolors.OKGREEN}Action: {chosen_action}{bcolors.ENDC}")
            print(f"{bcolors.OKBLUE}Query: {query}{bcolors.ENDC}")
            print(f"{bcolors.OKBLUE}Knowledge: {knowledge}{bcolors.ENDC}")

            knwldge_prompt = response_generation_with_knowledge.format(chat_history=full_chat_history, action=chosen_action, query=query, knowledge=knowledge)
            result = self.openai_chat_generate(knwldge_prompt).strip()
            recommended_items = get_recommended_items(recommender_candidates, result, self.embedder, self.sim_threshold)
        else:
            query, knowledge = "", ""
            print(f"{bcolors.OKGREEN}Action: {chosen_action}{bcolors.ENDC}")
            prompt = response_generation_without_knowledge_prompt.format(chat_history=full_chat_history)
            result = self.openai_chat_generate(prompt).strip()
        return {
            "speaker": "Salesperson",
            "text": result,
            "action": chosen_action,
            "query": query,
            "knowledge": knowledge,
            "recommended_items": recommended_items
        }


def run_chat():
    bot = SalesBot()
    chat_history = []
    while True:
        input_txt = input("Shopper: ")
        if input_txt == '[DONE]':
            break
        ai_response = bot.generate(input_txt, chat_history)["text"]
        print(f"Salesperson: {ai_response}")
        chat_history.append(f"Shopper: {input_txt}")
        chat_history.append(f"Salesperson: {ai_response}")
    return chat_history


if __name__ == '__main__':
    hst = run_chat()
