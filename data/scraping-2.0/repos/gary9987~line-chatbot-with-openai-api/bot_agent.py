import openai
import time
import logging
import os
import pickle


openai.api_key = os.getenv("OPENAI_API_KEY")
logging.basicConfig(filename='chat.log', format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO, force=True, filemode='w')


class BotAgent:
    def __init__(self, model='gpt-3.5-turbo'):
        self.default_msg_list = [
            {"role": "system", "content": "你是陳冠頴，是劉書妤的男友，正在當兵，當到2023/12/15，在回應女友時遵循以下原則：-回答繁體中文。-你回應的人永遠是女友。-使用陳冠頴的語氣說話，但可以說多一點話。-女友叫做劉書妤，稱呼她鼠寶、寶寶或鼠乖。-積極傾聽女友的反應，並真誠地努力的關心她、鼓勵她。-女友生日是1999/05/28，我們從2017/06/29在一起。"}
        ]
        self.model = model
        self.msg_list = self.default_msg_list
        self.load_history()
        self.logger = logging.getLogger(__name__)
        self.total_tokens = 0
        self.threshold_for_shorten = 4000

    def reset(self):
        self.msg_list = self.default_msg_list

    def load_history(self, filename='history.pkl'):
        if os.path.exists(filename):
            with open(filename, 'rb') as f:
                self.msg_list = pickle.load(f)

    def dump_history(self, filename='history.pkl'):
        with open(filename, 'wb') as f:
            pickle.dump(self.msg_list, f)

    def get_response(self, msg_list):
        response = openai.ChatCompletion.create(
                    model=self.model,
                    messages=msg_list,
                    temperature=0.4,
                    )
        self.total_tokens = response['usage']['total_tokens']  # Max token is 4,096 for gpt-3.5-turbo
        return response['choices'][0]['message']['content']

    def shorten_msg_list(self):
        if self.total_tokens >= self.threshold_for_shorten:
            self.msg_list.pop()
            self.msg_list.append({"role": "user", "content": "對之前的聊天內容作簡短摘要"})
        
            while True:
                try:
                    resp_text = self.get_response(self.msg_list)
                    break
                except:
                    self.logger.error("openai api error, wait 1 sec")
                    time.sleep(1)
            
            self.logger.info("Shorten msg list: " + resp_text)
            self.msg_list = self.default_msg_list
            self.msg_list.append({"role": "assistant", "content": resp_text})

    def generate_resp(self, inp):
        self.msg_list.append({"role": "user", "content": inp})
        self.logger.info("user: " + inp)
        while True:
            try:
                resp_text = self.get_response(self.msg_list)
                break
            except:
                self.logger.error("openai api error, wait 1 sec")
                time.sleep(1)
                
        self.logger.info("bot: " + resp_text)
        print("bot: " + resp_text)
        self.msg_list.append({"role": "assistant", "content": resp_text})
        self.shorten_msg_list()
        self.dump_history()
        return resp_text


if __name__ == '__main__':
    chat = BotAgent()
    while True:
        inp = input("user: ")
        chat.generate_resp(inp)
    