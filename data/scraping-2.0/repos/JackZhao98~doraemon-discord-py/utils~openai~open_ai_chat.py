import openai
import json
# import openai helper
# from utils.openai.openai_helper import create_chat_completion
from threading import Thread, Lock
import asyncio

class OpenAIChat:
    def __init__(self, api_key, org, model):
        self.api_key = api_key
        self.org = org
        self.model = model
        openai.api_key = api_key
        openai.organization = org
        self.messages={}
        self.dispatch_message = False

    def toJSON(self):
        return json.dumps(self, default=lambda o: o.__dict__, sort_keys=True, indent=4)

    def process_message(self, message, uid):
        if uid not in self.messages.keys():
            self.messages[uid] = []
        # if length of existing messages is greater than 10, remove first 2
        if len(self.messages[uid]) >= 10:
            self.messages[uid] = self.messages[uid][2:]
        self.messages[uid].append(message)

    def clear_message(self, uid):
        if uid in self.messages.keys():
            self.messages[uid].clear()

    def clear_all(self):
        self.messages.clear()

    def chat_stream(self, prompts):
        for chunk in openai.ChatCompletion.create(
            model=self.model,
            messages = prompts,
            stream=True,
        ):
            content = chunk.choices[0].get("delta", {}).get("content")
            # self.process_message(response_msg.toDict(), uid)
            if content is not None and content != "":
                yield content

    def update_message(self, lock, r, reply):
        with lock:
            r["message"] += reply

    def update_message_worker(self, lock, r, prompts):
        self.dispatch_message = True
        for reply in self.chat_stream(prompts):
            self.update_message(lock, r, reply)
        self.dispatch_message = False
        print ("done")

    async def chat_v2(self, user_prompt, uid, message_ctx):
        request = {"role": "user", "content": user_prompt}
        self.process_message(request, uid)
        r = {"message": ""}
        lock = Lock()
        try:
            t2 = Thread(target=self.update_message_worker, args=(lock, r, self.messages[uid]))
            t2.start()
            print()
            while self.dispatch_message == True:
                if r["message"] != "":
                    await message_ctx.edit(content=r["message"])
                    print(".", end="")
                await asyncio.sleep(0.6)
            # Finally, update the message
            await message_ctx.edit(content=r["message"])
            t2.join()
            self.process_message({"role": "assistant", "content": r["message"]}, uid)
            return ""
        except Exception as e:
            print(e)
            return "请使用 `-clear` 清除历史消息并重试。 Please try to use `-clear` to clear your chat history and try again."

    # Deprecated
    # def chat(self, message, uid):
    #     request = {"role": "user", "content": message}
    #     self.process_message(request, uid)

    #     try:
    #         chatResponse = self.generate_response(
    #             model_id=self.model,
    #             messages=self.messages[uid],
    #             temperature=0.9,
    #             max_tokens=2000,
    #             frequency_penalty=0.5,
    #             presence_penalty=0.6,
    #         )
    #     except openai.error.InvalidRequestError as e:
    #         print(e)
    #         return "请使用 `-clear` 清除历史消息并重试。 Please try to use `-clear` to clear your chat history and try again."

    #     if len(chatResponse.choices) == 0:
    #         return "请使用 `-clear` 清除历史消息并重试。 Please try to use `-clear` to clear your chat history and try again."
        
    #     response_msg = chatResponse.choices[0].message
    #     self.process_message(response_msg.toDict(), uid)
    #     return response_msg.content


    # Deprecated
    # def generate_response(self, model_id, messages, temperature=1, top_p=1, n=1, stream=False, stop=None, max_tokens=1000, presence_penalty=0, frequency_penalty=0,):
    #     response = openai.ChatCompletion.create(
    #         model=model_id,
    #         messages=messages,
    #         temperature=temperature,
    #         top_p=top_p,
    #         n=n,
    #         stream=stream,
    #         stop=stop,
    #         max_tokens=max_tokens,
    #         frequency_penalty=frequency_penalty,
    #         presence_penalty=presence_penalty,
    #     )
    #     response_str = json.dumps(response, indent=4)
    #     return create_chat_completion(response_str)
