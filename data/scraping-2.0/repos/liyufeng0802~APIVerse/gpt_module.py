import os
import yaml
import openai

class GPT_module:
    MODEL_NAME = "gpt-4"

    def __init__(self):
        openai.api_key = get_openai_api_key()

    def query(self, prompt):
        # 
        answer = self.comp(prompt, MaxToken=30, outputs=1)
        if len(answer) > 0:
            print(answer)
            return answer[0]['message']['content']
        else:
            print("[Error] No response from OpenAI model!")
            return None
    
    def comp(self, PROMPT, MaxToken=50, outputs=1): 
        response = openai.ChatCompletion.create(model=self.MODEL_NAME,
                                                messages=[{"role": "system", "content": 'You are a AI that understand all API technical concept'},
                                                {"role": "user", "content": PROMPT}])
        return response['choices']
    
    def query_with_conversation(self, prompt, conversation):
        """"
        GPT query in a conversation format
        conversation:
        [
            {"role": "system", "content": "You are a coding tutor bot to help user write and optimize python code."},
            {"role": "user", "content": "Brabra"}
            {"role": "assistant", "content": "Brabra"}
        ]
        (Note: remeber to keep increase the conversation)

        Output:
        reponse: the response for this prompt
        conversation: update conversation
        """
        conversation.append({"role": "user", "content": prompt})
        response = openai.ChatCompletion.create(
            model=self.MODEL_NAME,
            messages=conversation
        )
        conversation.append({"role": "assistant", "content": response["choices"][0]["message"]['content']})
        response_message = response["choices"][0]["message"]
        return response_message, conversation

# Read OpenAI key from ~/.cal_hack/config.yaml
def get_openai_api_key():
    with open(os.path.expanduser("~/.cal_hack/config.yaml"), "r") as f:
        config = yaml.load(f, Loader=yaml.SafeLoader)
        if "openai_api_key" not in config:
            raise Exception("OpenAI API key not found in ~/.cal_hack/config.yaml,"
                            " please add 'openai_api_key: YOUR_KEY' in the file.")
        openai_api_key = config["openai_api_key"]
        return openai_api_key