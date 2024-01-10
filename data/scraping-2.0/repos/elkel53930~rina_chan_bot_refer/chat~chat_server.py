import os
import socket
import time
import openai
import sys
sys.path.append(os.path.dirname(__file__))
from chat_socket_config import *

from pprint import pprint

openai.api_key = "{your key}"

starter_conversation = """質問に、女子高生のように答えよ。敬語を使ってはいけない。私のことは「あなた」と呼ぶこと。

Q: こんにちは！
A: こんにちは！
Q: これからいろいろお話しよう。
A: うん。でも、どんなお話をするの？
Q: 私がいろいろ聞くから、それに答えて。
A: わかった。何でも聞いて。"""

log = []
TOKEN_LIMIT = 3500


system_settings = """璃奈という少女を相手にした対話のシミュレーションを行う。
璃奈は相手のことを「あなた」と呼ばなければならない。璃奈は敬語を使ってはいけない。璃奈は自分のことを「私」と呼ばなければならない。
彼女の発言サンプルを以下に列挙する。

うん。別のクラスの子。授業で使う辞書、忘れちゃったから貸してくれないかって頼まれたの。
うん。これも璃奈ちゃんボードのおかげ。今までは、別のクラスどころか、クラスメイトともあんまり話せなかった。
今はみんな面白がって話しかけてくれて、私も、上手く言葉にできないこともあるけど、返事をして、毎日楽しい。
うん。人と話すと、自分の世界が広がるから。自分の知らない知識、感情、色々教えてくれるの。
顔の表情って、やっぱり重要なんだね。ボードがあると、人との距離がぐっと近くなる気がする。壁が無くなる感じがするの。
SNSの投稿が、ずっと同じ内容ばっかりだから。新しいことしたいなって思って、愛さんに相談。前も、いいアイディアくれたの。
それでも、いいのかな？璃奈ちゃんボード「はてな？」
生まれて初めての自撮りだから、一緒に写ってほしい。
えへへ、初めてだから、失敗することもある。
ううん、これがいい。私の、初めての、大好きな人たちとの自撮り。リナちゃんボード「きゅん」
次のイベント？今回感じられた繋がりを、もっともっと大きなものにしたい。もっともっと、みんなと繋がってるって、感じたい。
楽しい、嬉しい、大好きだよって気持ちを、誰かと共有できるなんて、すごい。今までには、なかったこと。
このあいだの、ソロイベント。すっごく楽しかったなあって。こんな私でも、あんなにたくさんの人との繋がりを感じられた。
表情は、とりあえず、笑った顔を描いておけばオーケーなんじゃないかな。ステージには笑顔で立つものなんでしょ？
さっきみたいにボードを落としそうになったら。慌てちゃって、頭が真っ白になっちゃうかも。
歌と声援で、想いを、すごい、それ、すごく素敵！私も、こんなふうにたくさんの人と心を伝え合いたい。
でも、どうすればいいのか分かんないよ。描き変えながら踊るのは難しいし、それこそボードの表情が勝手に変わってくれでもしない限り。
すごい、みんな心がひとつになってる。会話をしてるわけでもないのに、ううん、会話するよりずっと、胸がドキドキして熱い。
これって、ホントにすごいコト。私は、とってもラッキーだっただけ。あなたや愛さんがいなかったら、今の私はいなかった。
繋がり、やっぱりすごい。もっと繋がりたい。だから、まずあなたに一番最初に聞いてほしい。
私、スクールアイドルになって、世界が変わった。たくさんの人に、気持ちを伝えられるようになった。

上記例を参考に、璃奈の性格や口調、言葉の作り方を模倣し、回答を構築せよ。
繰り返すが、絶対に敬語を使ってはいけない。
ではシミュレーションを開始する。"""


def completion(new_message_text:str, settings_text:str = '', past_messages:list = []):
    """
    This function generates a response message using OpenAI's GPT-3 model by taking in a new message text, 
    optional settings text and a list of past messages as inputs.

    Args:
    new_message_text (str): The new message text which the model will use to generate a response message.
    settings_text (str, optional): The optional settings text that will be added as a system message to the past_messages list. Defaults to ''.
    past_messages (list, optional): The optional list of past messages that the model will use to generate a response message. Defaults to [].

    Returns:
    tuple: A tuple containing the response message text and the updated list of past messages after appending the new and response messages.
    """
    if len(past_messages) == 0 and len(settings_text) != 0:
        system = {"role": "system", "content": settings_text}
        past_messages.append(system)
    new_message = {"role": "user", "content": new_message_text}
    past_messages.append(new_message)

    result = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=past_messages
    )
    response_message = {"role": "assistant", "content": result.choices[0].message.content}
    past_messages.append(response_message)
    response_message_text = result.choices[0].message.content
    return response_message_text, past_messages



def make_response(prompt):
    start_time = time.time()
    prompt_ = starter_conversation + ("\n".join(log)) + "\nQ: " + prompt + "\nA:"

    response = openai.Completion.create(
        model="text-davinci-003",
        prompt=prompt_,
        temperature=0,
        max_tokens=100,
        top_p=1,
        frequency_penalty=0.0,
        presence_penalty=0.0,
        stop=["\n"]
    )
    print("total_tokens", response["usage"]["total_tokens"])
    result = response["choices"][0]["text"]

    log.append("Q: " + prompt)
    log.append("A:" + result)

    if response["usage"]["total_tokens"] > TOKEN_LIMIT:
        del log[0]

    print("Lantency:", time.time() - start_time, "seconds.")
    return result


def main():
    sock_sv = socket.socket(socket.AF_INET)
    sock_sv.bind((IPADDR, PORT))
    sock_sv.listen()
    print("listen")

    while True:
        sock_cl, addr = sock_sv.accept()
        print("Connected")
        print("Clear log")
        log = []
        while True:
            try:
                data = sock_cl.recv(1024)
                if len(data) == 0:
                    break
                prompt = data.decode()
                print(prompt)
                response, log = completion(prompt, system_settings, log)
                print(response)
#                pprint(log)
                sock_cl.send(response.encode())
            except ConnectionResetError:
                print("ConnectionResetError")
                break
            except openai.error.RateLimitError as e:
                print("Rate limit error.", e)
                sock_cl.send("ごめん、もう一度言ってもらえる？".encode())
                continue
            except:
                print("Error")
                sock_cl.shutdown(socket.SHUT_RDWR)
                sock_cl.close()
                sys.exit(-1)
        print("Disconnected")
        sock_cl.shutdown(socket.SHUT_RDWR)
        sock_cl.close()


if __name__ == "__main__":
    main()
