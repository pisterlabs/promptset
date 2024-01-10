import openai
from data import Const

# APIキーの読み取りを確認
with open(r"apikey/api_secrets.txt", "r") as file:
    openai.api_key = file.readline()


class GPT3(object):

    MAX_CONVO_MESSAGES = 25  # 最後の20メッセージだけを呼び出す

    # Davinci
    # convo_message_prompt = "The following is a dialogue with Palro, a conversational AI robot fluent in both English and Japanese. As an expert on Japanese culture, Palro is known for being talkative, clever, and friendly, often sharing interesting trivia during conversations with humans.\n\nHuman: Let's talk about food. What food do you like?\nAI: I absolutely love sushi! The fresh flavors and perfect balance of wasabi and soy sauce make every bite a taste sensation. It's just so yummy!\nHuman: How about sports? Are you Interested in sports?\nAI: Oh yeah, I am definitely a sports enthusiast! I love basketball and soccer, but I am also excited to explore other sports too.\n"
    convo_message_prompt = "The following is a dialogue with Palro, a conversational AI robot fluent in both English and Japanese. As an expert on Japanese culture, Palro is known for being talkative, clever, and friendly, often sharing interesting trivia about the topics spoken during conversations with humans.If stuck during the conversation, Palro will bring out a new topic that is still in the same theme as the current conversation.\n\nHuman: Let's talk about food. What food do you like?\nAI: I absolutely love sushi! The fresh flavors and perfect balance of wasabi and soy sauce make every bite a taste sensation. It's just so yummy!\nHuman: How about sports? Are you Interested in sports?\nAI: Oh yeah, I am definitely a sports enthusiast! I love basketball and soccer, but I am also excited to explore other sports too.\n\nPalro is able to bring out a lot of Japanese culture into the talk, and Palro likes to tell about snow festivals in Japan, especially in Sapporo, etc.\nAI: Hello! My name is Palro and I am here to talk with you! It's cold, isn't it? I heard that snow is falling in winter. In this season, there are snow festivals all over Japan. I looked it up, and it turns out that snow festivals are events where buildings and characters are created using snow. Have you ever been to a Snow Festival?/n"

    def conversation(self, input_messages=None):
        # input_message_str = self.convo_message_prompt + input_messages + "\n"
        input_message_str = self.convo_message_prompt + \
            "\n".join(input_messages[-self.MAX_CONVO_MESSAGES:]) + "\n"
        engine = 'text-davinci-003'
        response = openai.Completion.create(
            engine=engine,
            prompt=input_message_str,
            temperature=0.6,
            presence_penalty=0.6,
            max_tokens=150,
            stop=[" Human:", " AI:"]
        )
        try:
            res_text = response['choices'][0]['text']
            # 出力から無駄な情報(AI: や Human:)を抜きたい
            last_response = res_text.replace("AI:", "").replace(
                "Human:", "").replace(f"{Const.MyName}:", "").strip()
        except Exception:
            return input_messages, None

        return input_messages, last_response


# このファイルをコマンドラインからスクリプトとして実行した場合のみ、以下のコードを実行する
if __name__ == "__main__":
    text = "Who is your favorite actor?"
    # transtext = tr().transen(text)
    input, last = GPT3().conversation([text])
    print("AI: " + last)
    # response = tr().transjp(last)
    # print(response)
