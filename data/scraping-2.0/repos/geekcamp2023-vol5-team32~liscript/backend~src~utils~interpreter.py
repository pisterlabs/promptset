import json
import os

import langid
import openai
from dotenv import load_dotenv

load_dotenv(verbose=True)
openai.api_key = os.getenv("OPENAI_API_KEY")

class Writer():
    def __init__(self, original_text:str):
        self.original_text = original_text
        self.model_of_ChatGPT = "gpt-3.5-turbo"
        self.given_msg = {
            "role": "user", "content": self.original_text
        }

    def __repr__(self):
        return '\n'.join([
            f'Original text: {self.original_text}',
            f'ChatGPT mode: {self.model_of_ChatGPT}',
        ])

    def __callChatGPT(self, msgs:list) -> str:
        gptRepr = openai.ChatCompletion.create(model=self.model_of_ChatGPT, messages = msgs)
        extractRepr = lambda repr: repr["choices"][0]["message"]["content"]
        return extractRepr(gptRepr)

    def __callStreamChatGPT(self, msgs:list):
        gptGen = openai.ChatCompletion.create(model=self.model_of_ChatGPT, messages = msgs, stream=True)
        extractChunk = lambda chunk: chunk["choices"][0].get("delta",{}).get("content")
        for chunk in gptGen:
            word = extractChunk(chunk)
            if word is not None:
                yield word.encode('utf-8')

    def translatorGPT(self, target_lang="", stream=False):
        code = F"{target_lang}" if target_lang!="" else "ja"
        basedir = os.path.dirname(__file__)
        jsonpath = os.path.join(basedir, "iso_639-1.json")
        with open(jsonpath) as f:
            language_map = json.load(f)
        translator_role_msg = {
            "role": "system",
            "content": F"あなたは与えられたテキストを{language_map[code]['nativeName']}に翻訳しなければならない。"
        }
        msg = [translator_role_msg, self.given_msg]
        if (stream):
            gpt_answer_gen = self.__callStreamChatGPT(msg)
            return gpt_answer_gen
        else:
            gpt_answer = self.__callChatGPT(msg)
            return gpt_answer

    def summarizerGPT(self, stream=False):
        summarizer_role_msg = {
            "role": "system",
            "content": "あなたは与えられたテキストを要約しなければならない。要約結果はそのまま出力する。"
        }
        msg = [summarizer_role_msg, self.given_msg]
        if (stream):
            gpt_answer_gen = self.__callStreamChatGPT(msg)
            return gpt_answer_gen
        else:
            gpt_answer = self.__callChatGPT(msg)
            return gpt_answer

def linguist(unknown_language_text:str) -> str:
    language_codes, _ = langid.classify(unknown_language_text)
    return language_codes

if __name__=="__main__":
    import time
    writer = Writer("Red sky at night, sailors' delight.Red sky at morning, sailors take warning.")
    start = time.time()
    result = writer.translatorGPT(target_lang="ja", stream=True)
    for i in result:
        if (i == None or i == ""):
            continue
        else:
            print(i)
    result = writer.summarizerGPT(stream=True)
    end = time.time()
    print(F"runtime:{end-start}")
    start = time.time()
    for i in result:
        if (i == None or i == ""):
            continue
        else:
            print(i)
    end = time.time()
    print(F"runtime:{end-start}")

# result
# This is a story from when I was living alone in Aichi Prefecture, away from my parents when I entered university. In the spring of my second year, some friends and I decided to get our driver's licenses together during summer vacation. I remember there was a course that would be slightly cheaper if a few people applied together. At that time, I was living off my parents' allowance and working part-time at a movie theater. But that wasn't enough to cover the cost of the driving school, so I reluctantly started a new part-time job at a lounge near my school. The lounge wasn't as fancy as a cabaret club in the downtown area, in terms of dress code and the level of the girls. Even a country bumpkin like me was able to easily pass the interview. After working at the lounge for about a week, I was assigned to assist Mr. Makabe, a customer. Mr. Makabe was a man in his 50s, with thinning hair and a sturdy build. He was your typical ordinary middle-aged man, but apparently he was a company executive who would come in once a week to have a few drinks and leave promptly. He seemed to be a regular customer, as the girls and the boys referred to him as "Maa-chan" or "Maa-san." Before I sat down at his table, the boy who showed me to the seat told me, "He drinks elegantly and is a kind person. You're lucky." However, as soon as Mr. Makabe saw me, he made a shocked face and exclaimed loudly, "Wow, I can't believe it. You're really not my type!" It annoyed me internally, but I had become used to being criticized about my appearance since starting the job. I served him without caring and he kept repeating, "You're not my type." I was genuinely puzzled by his reaction and asked for help from a senior cast member who had been assigned to him, but she looked pale and acted suspiciously. In the end, she left work early due to feeling unwell after Mr. Makabe left and quit the job soon after. A few days later, Mr. Makabe came to the lounge and requested for me, even though he had previously said I wasn't his type. He still muttered, "You're not my type," in between conversations, but he didn't bother me too much and he gave me plenty of drinks, just as the boy had said. I felt that he was a good person. Some time later, Mr. Makabe invited me to go out with him. Going out with a customer involved having a meal and spending some time together before going to work, and it was allowed by the establishment. I decided to accept his invitation, and we went to a kushikatsu (fried skewers) restaurant a little far from the club to have dinner. On the day of our outing, we took a taxi that Mr. Makabe called to take us back to the club after our meal. I sat on the left side of the back seat, while Mr. Makabe sat on the right side. On the way to the club, Mr. Makabe pointed to big houses and apartment buildings and said, "That's my place," making silly jokes. While I casually continued the conversation, I suddenly heard a voice from the left side during a traffic light stop saying, "This place is mine..." The voice was too soft to hear the rest clearly, but I thought Mr. Makabe was joking again and turned to him, saying, "Oh, not this again?" However, Mr. Makabe had not said anything. He was staring out of the window behind me expressionlessly. When I was drawn to look outside, I saw a man standing in the cemetery on the side of the road. Our eyes met, and when the man moved his lips, I heard him say, "This place is mine..." Even though there was a sidewalk between us, and the car windows were closed, I heard it clearly. A chilling sensation ran down my spine, and I quickly averted my gaze from the graveyard. Nevertheless, I could still hear the man's voice. Eventually, we arrived at the club and the voice stopped as I got out of the taxi. Mr. Makabe went ahead to his seat, and I finished changing and headed to my seat a little later. When I tried to ask Mr. Makabe about what had just happened, I heard the voice say, "This place is mine..." again. Mr. Makabe laughed and said, "Oh, you've encountered it too." I couldn't understand why he was able to laugh in this situation or if he could even hear the voice, and my mind was filled with various questions. As I was lost in thought, I saw the man standing at the edge of my field of vision. He stood next to the wall, staring intently at me. I realized that he was the same man from the cemetery. Nobody else, including the boy and other cast members, seemed to see him, as nobody reacted. The man slowly glided towards us without moving his feet and stopped right next to our table. He leaned his face close and whispered, "This place is mine..." I was terrified and on the verge of tears, but Mr. Makabe didn't seem to care about the man and said with a smirk, "If you're scared, you can hold onto my arm." In a mix of fear and anger, my emotions were in disarray, but I had no choice. I desperately held onto Mr. Makabe and turned to him with twice the energy to continue the conversation. The man's voice gradually became clearer, and it felt like I would hear the rest of his sentence if I let my guard down. What would I do once Mr. Makabe left? I didn't think I could endure it alone. I would have to go for a purification ceremony. While I was thinking about various things, unfortunately, it was time for Mr. Makabe to leave. As I got up to see him off, the presence of the man suddenly disappeared. I instinctively looked where the man had been, but there was nothing there. Puzz
# runtime:40.92913269996643
# 私は大学進学を機に愛知県で一人暮らしを始めた際、学校近くのラウンジでバイトを始めました。そこで真壁さんというお客さんに出会い、彼は霊感が強く、私にも見える存在がいることが判明しました。バイト中に真壁さんと一緒にタクシーに乗ると、窓の外に男性の姿が現れ、「ここ、俺の…」と呟いていました。怖がる私に対しても真壁さんは冗談めかして接し、お店に着くと男の姿は消えました。後日真壁さんに聞いたところ、彼は霊を見ることができ、私が見た男性は焼死体のような存在だったそうです。私は車校代を貯めるためにバイトを辞めましたが、真壁さんとは今でも連絡を取り合っています。また、彼は別のキャストにも霊を見せることができるため、今はその子を指名するようになったと言っています。
# runtime:11.955734252929688