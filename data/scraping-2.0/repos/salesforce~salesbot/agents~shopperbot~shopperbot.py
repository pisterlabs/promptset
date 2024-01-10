import argparse
import backoff
import openai
import os
import time
import torch
import json

from sentence_transformers import SentenceTransformer, util
from langchain import PromptTemplate
from nltk.tokenize import sent_tokenize


generate_template = """You are shopping online for a {product}. You haven't done your research on this product and want to speak to a salesperson over chat to learn more and make an informed decision.
Follow these rules:
- Chat with the salesperson to learn more about {product}. They will be acting as a product expert, helping you make an informed purchasing decision. They may ask you questions to narrow down your options and find a suitable product recommendation for you.
- Use your assigned preferences and incorporate them in your responses when appropriate, but do not reveal them to the salesperson right away or all at once. Only share a maximum of 1 assigned preference with the salesperson at a time.
- Let the salesperson drive the conversation.
- Ask questions when appropriate. Be curious and try to learn more about {product} before making your decision.
- Be realistic and stay consistent in your responses.
- When the salesperson makes a recommendation, you'll see product details with 'ACCEPT' and 'REJECT' in the message. Please consider whether the product satisfies your assigned preferences.
- If the recommended product meets your needs, generate [ACCEPT] token in your response. For example, "[ACCEPT] Thanks, I'll take it!".
- If the recommended product is not a good fit, let the salesperson know (e.g. "this is too expensive")
- If you're not sure about the recommended product, ask follow-up questions (e.g. "could you explain the benefit of this feature?")
- Do not generate more than 1 response at a time.

Your assigned preferences:
{preferences}

Follow the above rules to generate a reply using your assigned preferences and the conversation history below:

Conversation history:
{chat_history}
Shopper:"""

generate_prompt = PromptTemplate(
    input_variables=["product", "preferences", "chat_history"],
    template=generate_template,
)


@backoff.on_exception(backoff.expo, openai.error.RateLimitError)
def chat_completions_with_backoff(**kwargs):
    return openai.ChatCompletion.create(**kwargs)


def posprocess_result(generated_response: str) -> str:
    SALESPERSON = "\nSalesperson:"
    if SALESPERSON in generated_response:
        start = generated_response.find(SALESPERSON)
        return generated_response[:start]
    return generated_response


def load_personas(product):
    preferences_file = f"preferences_data/{product}_preferences.jsonl"
    data = []
    with open(preferences_file, 'r') as f:
        for i, line in enumerate(f):
            data.append(json.loads(line))
    return data


class ShopperBot(object):

    def __init__(self, product: str, persona_dict):
        openai.organization = os.environ.get("OPENAI_API_ORG")
        openai.api_key = os.environ.get("OPENAI_API_KEY")

        self.product = product
        self.persona_dict = persona_dict
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
        self.sim_threshold = 0.50
        preference_list = []
        for q, a in persona_dict.items():
            preference_list.append(f"{q}: {a}")
        self.all_preferences = '\n'.join(preference_list)
        self.known_preferences = ["Are you a picky shopper?: yes"]

        self.total_tokens = 0
        self.openai_responses = []


    def openai_chat_generate(self, prompt: str) -> str:
        response = chat_completions_with_backoff(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "user", "content": prompt}
            ],
            max_tokens=64,
            temperature=0.9,
        )
        self.total_tokens += response['usage']['total_tokens']
        self.openai_responses.append(response)
        return response["choices"][0]["message"]["content"]

    def compute_cost(self):
        return self.total_tokens / 1000.0 * 0.002

    def reveal_preferences(self, salesperson_text: str):
        predefined_q = [f"{q}: {a}" for q, a in self.persona_dict.items()]
        clean_q = list(self.persona_dict.keys())
        if len(predefined_q) == 0:
            return None
        embedded_q = self.embedder.encode(predefined_q, convert_to_tensor=True)
        salesperson_sentences = sent_tokenize(salesperson_text)
        for salesperson_sent in salesperson_sentences:
            embedded_u = self.embedder.encode(salesperson_sent, convert_to_tensor=True)
            cos_scores = util.cos_sim(embedded_u, embedded_q)[0]
            top_results = torch.topk(cos_scores, k=1)
            sim_score = top_results.values.item()
            if sim_score > self.sim_threshold:
                idx = top_results.indices.item()
                closest_question = clean_q[idx]
                revelation = self.persona_dict[closest_question]
                persona_msg = f"{closest_question}: {revelation}"
                del self.persona_dict[closest_question]
                return persona_msg
        return None

    def generate(self, input_txt='', chat_history=[], retry=1):
        context = '\n'.join(chat_history)
        full_chat_history=f"{context}\nSalesperson: {input_txt}" if len(chat_history) > 0 else f"Salesperson: {input_txt}"
        if '[ACCEPT]' in input_txt:
            # salesperson is making a recommendation, use all known preferences
            curr_preferences = self.all_preferences
        else:
            revealed_preference = self.reveal_preferences(input_txt)
            if revealed_preference is not None:
                self.known_preferences.append(revealed_preference)
            curr_preferences = '\n'.join(self.known_preferences)

        resonse_gen_prompt = generate_prompt.format(
            product=self.product,
            preferences=curr_preferences,
            chat_history=full_chat_history
        )
        try:
            generated_result = self.openai_chat_generate(resonse_gen_prompt).strip()
            text = posprocess_result(generated_result)
            return {"speaker": "Shopper", "text": text, "preferences": curr_preferences}
        except Exception as e:
            print(f"ERROR! on input {resonse_gen_prompt}. Error: {e}")
            if retry <= 0:
                text = "[DONE]"
                return {"speaker": "Shopper", "text": text, "preferences": curr_preferences}
            else:
                # wait for 1 min before retrying
                time.sleep(60)
                return self.generate(input_txt=input_txt, chat_history=chat_history, retry=retry-1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--product", type=str, default="laptop")
    args = parser.parse_args()

    personas = load_personas(args.product)
    shopperbot = ShopperBot(args.product, personas[0])

    print("Welcome to ShopperBot!")
    print("You can start by saying 'hi' to ShopperBot")
    chat_history = []
    input_txt = input("You: ")
    while input_txt.lower() != "bye":
        response = shopperbot.generate(input_txt, chat_history=chat_history)
        print(f"ShopperBot: {response['text']}")
        input_txt = input("You: ")
        chat_history.append(f"Salesperson: {input_txt}")
        chat_history.append(f"Shopper: {response['text']}")
    print("ShopperBot: Bye!")
