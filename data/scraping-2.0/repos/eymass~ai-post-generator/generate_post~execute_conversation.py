import json
import openai
import time

from generate_post.prepare_prompt import PromptGenerator


class ConversationExecutor:
    def __init__(self, api_key, products_count=5):
        self.api_key = api_key
        self.conversation = []
        self.products_count = products_count
        self.num_of_requests = 0

    def generate_post_json(self, topic):
        p_gen = PromptGenerator(topic)
        first_prompt = p_gen.get_prompt()
        print(first_prompt)
        self.conversation = [
            {"role": "system", "content": "You are a female beauty products "
                                          "promoter using blog posts, who is friendly, professional and excited."},
            {"role": "user", "content": first_prompt},
        ]

        # introduction
        post_str = self.send_message()
        self.add_assistant_to_conversation(post_str)
        print(post_str)
        # get dict
        post_dict = self.json_str_2_dict(post_str)

        # products
        products = []
        for i in range(self.products_count):
            self.conversation.append({"role": "user",
                                      "content": p_gen.get_product_prompt(i)})
            product_str = self.send_message()
            self.add_assistant_to_conversation(product_str)
            print(product_str)

            product_dict = self.json_str_2_dict(product_str)
            product_dict['images'][0]['src'] = product_dict['key']
            products.append(product_dict)

        post_dict["products"] = products
        # comments
        self.conversation.append({"role": "user",
                                  "content": p_gen.get_comments_prompt(4)})
        comments = self.send_message()
        print(comments)
        comments_dict = self.json_str_2_dict(comments)
        post_dict["comments"] = comments_dict
        print(f"[generate_post_json] returning post_dict")
        self.num_of_requests = 0
        return post_dict

    def send_message(self):
        if self.num_of_requests >= 3:
            print(f"[send_message] reached rate limit of 3, wait")
            time.sleep(70)
            self.num_of_requests = 0
        print(f"[send_message] continue")
        self.num_of_requests += 1
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=self.conversation,
            api_key=self.api_key
        )
        return self.parse_chat_response(response)

    def parse_chat_response(self, response):
        return response['choices'][0]['message']['content']

    def json_str_2_dict(self, json_text):
        # Remove leading and trailing whitespaces
        json_text = json_text.strip()

        # Remove potential escape characters
        json_text = json_text.replace('\\"', '"')

        return json.loads(json_text)

    def add_assistant_to_conversation(self, post_json):
        # Update the conversation with the new post JSON
        self.conversation.append({"role": "assistant", "content": post_json})

    def str_to_dict(self, str_json):
        return json.loads(str_json)
