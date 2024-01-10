# %%
from langchain.llms import GPT4All
#from models.model_cn import ChatGLM

from friend_replica.format_chat import ChatConfig, format_chat_history
from friend_replica.recollection import LanguageModelwithRecollection
from friend_replica.semantic_search import *


# %%
### Main
# Initialize Chat with one friend
chat_config = ChatConfig(
    my_name="Rosie",
    friend_name="Enri",
    language="english",
)
chat_with_friend = Chat(device='mps', chat_config=chat_config)
chat_blocks = chat_with_friend.chat_blocks
print([len(c) for c in chat_blocks])

'''
Example Output (English):

chat_Enri vectorized
[4, 11, 4, 3, 5, 27, 12, 14, 5, 17]
'''


# %%
# Semantic Memory Search among chat history with this friend
queries = ["sad"]
print("Searching for:", queries)
contexts = chat_with_friend.semantic_search(queries)
for context in contexts:
    print('\n'.join(format_chat_history(context, chat_with_friend.chat_config, for_read=True, time=True)))
    print()

'''
Example Output (English):

Searching for: ['sad']
2023-08-31T22:40, Rosie: I ruined my day with that pack of junk food tho [Sob]
2023-08-31T22:40, Andrew: (Sent a sticker)
2023-08-31T22:41, Rosie: Woke up at 8, did core exercise, studied, did hip exercise, studied then finally chips wtf
2023-08-31T22:41, Andrew: Wtf 🤷‍♂️🤷‍♂️🤷‍♂️
2023-08-31T22:41, Andrew: You were in a such good combo
2023-08-31T22:41, Andrew: And
2023-08-31T22:41, Andrew: Ruined it …
2023-08-31T22:41, Andrew: Hope it was good chips 
2023-08-31T22:42, Andrew: Not the shitty Lays 🫠
2023-08-31T22:42, Andrew: And 
2023-08-31T22:42, Andrew: Not a fucked up flavor 🫠
2023-08-31T22:42, Andrew:  If you dare telling me
2023-08-31T22:42, Andrew: It was Lays with Seaweed flavor…
2023-08-31T22:43, Andrew: (Sent a sticker)
2023-08-31T22:56, Rosie: no it’s not even real chips
2023-08-31T22:57, Rosie: (Sent an image)
2023-08-31T23:00, Andrew: (Sent a sticker)
2023-08-31T23:00, Andrew: Nooooooo

'''


# %%
# Load Memory Recollection Model
model = GPT4All(model="llama-2-7b-chat.ggmlv3.q4_0.bin", allow_download=False)
# model = ChatGLM()
m = LanguageModelwithRecollection(model, chat_with_friend)

# %%
# Memory Archive Generation
memory_archive  = m.memory_archive()

'''
Example Output (English):
####### Memory entry from 2023-08-07 22:59 to 2023-08-09 02:51: 
Memory:   The conversation is about a person named Enri who wants to use Rosie's VPN for their Apple device they bought in Venice, Italy. Rosie suggests that if it's an Android phone bought in China, there might be more problems with that. Enri offers to share their online video titled "August Travel Tips for People Who Want to Go to China Together" with Rosie in exchange for Rosie's VPN secrets.
Key Word:  Apple device + Venice + Italy
######## 
####### Memory entry from 2023-08-29 14:49 to 2023-08-31 14:25: 
Memory:   Rosie and Enri had a conversation about memes, with Enri sharing a video titled "#memes #黑人 #Chrishere #英语单词 #搞笑" and Rosie finding it funny.
Key Word:  Memes
######## 
####### Memory entry from 2023-08-31 14:25 to 2023-08-31 18:48: 
Memory:   Enri is busy with school work and has a presentation on short notice, while Rosie makes fun of them for being "racist" and "arrogant."
Key Word:  Busy student faces criticism
######## 
####### Memory entry from 2023-10-04 08:54 to 2023-10-06 22:30: 
Memory:   Rosie and Enri are having a conversation on WeChat. They discuss their experiences with Python for data analysis, regression, and difference in difference. Enri asks about using Python instead of specialized software like Stata, and Rosie replies that she uses R Studio for her stats project because it has a perfect interface. They also talk about going to office hours and looking for friends on a forum called "Popi" (which is written in Chinese characters). Enri mentions that he cannot attend office hours due to difficulty, and Rosie jokes that they could rent a fake alibi. The conversation ends with them saying goodbye and expressing their excitement for the weekend.
Key Word: "Python stats discussion"
######## 
####### Memory entry from 2023-10-06 23:27 to 2023-10-08 00:11: 
...
Memory:   Rosie and Enri are discussing a French guy that Rosie met online. Rosie is not interested in meeting up with him due to his attitude towards her, and Enri agrees with her assessment of French guys being shitty. They make jokes about the situation and reaffirm their friendship despite any negative experiences with French people.
Key Word:  Rosie & Enri discuss French guy
######## 
######## Finished Memory Archive Initialization of friend 'Enri'
'''


# %%
# Memory summary for one chat block
print('\n'.join(format_chat_history(chat_blocks[4], chat_with_friend.chat_config, for_read=True, time=True)))
print()
summary = m.summarize_memory(chat_blocks[4])
print(summary)
topic = m.generate_thoughts(summary, key_word_only=True)
print(topic)

'''
Example Output (English):
2023-08-11T05:09, Eddie: (Sent the link of an online video titled 'China is really out of this world ❤️ 🥵')
2023-08-11T11:45, Rosie: I’ve never heard about the places in this video hahahah but let’s go Dunhuang maybe (Sent an image) (Sent an image) You could ride camel and see these cave arts  (Sent an image) I’m bored. How’s the place you’re traveling at? Send me some pics 
2023-08-11T15:17, Eddie: Let’s absolutely go to Dunhuang When would be the best period?  (Sent an image) (Sent an image) (Sent a video) (Sent an image)
2023-08-11T15:32, Rosie: Peaceful village 
2023-08-12T08:13, Eddie: Not very hahaha They were shooting fireworks every night lol
2023-08-12T10:43, Rosie: wow quite romantic and good for couple travelling hahahah, have fun bb
2023-08-13T00:23, Eddie: Love u thx ❤️ (Sent the link of an online video titled '你都去过哪里呢😍 #旅行 #爱中国')
2023-08-13T00:27, Rosie: hahahha I could see you can’t wait to travel here Be sure to not do it during the October National Day holiday tho  It would be freaking crowded everywhere 
2023-08-13T03:21, Eddie: I can’t wait you’re right haha I’m gonna chill in october i guess Visit beijing likely
 
Rosie and Eddie are discussing travel destinations, with Eddie expressing interest in visiting Dunhuang, while Rosie recommends avoiding the October National Day holiday due to crowds. 
They also share images and videos of their respective locations, with Eddie looking forward to traveling in China and Rosie mentioning that she can't wait to see Eddie's adventures.
'''


# %%
# Personality Archive Generation
personality_archive = m.personality_archive()

'''
Example Output (English):
######## Personality entry from 2023-08-07 22:59 to 2023-08-09 02:51:
 Rosie is a tech-savvy and mischievous person who enjoys sharing tips and tricks, while Enri is a lighthearted and playful individual who is willing to share their travel recommendations.
######## Personality entry from 2023-08-10 08:42 to 2023-08-13 03:21:
 Rosie is a travel-enthusiast who has been to various places in China, including Dunhuang. She provides recommendations and tips for Enri, who is planning to visit China soon. Rosie is bubbly and enthusiastic about traveling, often using emojis and GIFs in her responses. Enri seems to be excited about the trip but also aware of the crowds during the National Day holiday in October.
...
######## Personality entry from 2023-10-30 21:03 to 2023-10-30 21:09:
 Rosie is a fan of the tattoo artist and wants to get a tattoo from her in Chengdu, but Enri is advising her not to overspend. The two have a playful and lighthearted relationship, with Enri using humor to try to calm Rosie's excitement about getting a tattoo.
######## Personality entry from 2023-11-08 18:39 to 2023-11-08 19:09:
 Rosie has a negative view of French guys and had a bad experience with one person in particular, who deleted her after she declined his invitation to dinner. She is frustrated that he only has Wednesdays off and won't compromise on their planned meeting time. Enri shares her frustration and reassures her that they are friends and will support her.
######## Finished Personality Archive Initialization of friend 'Enri'
'''

# %%
# Personality and Relationship Summary with one chat block
print('\n'.join(format_chat_history(chat_blocks[4], chat_with_friend.chat_config, for_read=True, time=True)))
print()
personality = m.generalize_personality(chat_blocks[4])
print(summary)
'''
Example Output (English):
Rosie is a fun-loving and adventurous person who enjoys traveling and exploring new places. 
Eddie and Rosie have a friendly and casual relationship, with Eddie seeking advice from Rosie on their shared interest in traveling in China.
'''


# %%
# Chatbot Friend Replica
print(m.chat_with_archive())

'''
Example Ouput (English):
Hi, Enri! I'm the agent bot of Rosie. I have memory of us discussing these topics:
#0 08.09:  Apple device + Venice + Italy
#1 08.13:  Travel China VPN
#2 08.17:  Travel plans & VPN sharing
#3 08.31:  Memes
#4 08.31:  Busy student faces criticism
#5 10.06:  Python stats discussion
#6 10.08:  Club plans
#7 10.24:  Social media conversation
#8 10.30:  Tattoos
#9 11.08:  Rosie & Enri discuss French guy
Do you want to continue on any of these?
Okay! Let's continue on [ Memes]
I recall last time:   Rosie and Enri had a conversation about memes, with Enri sharing a video titled "#memes #黑人 #Chrishere #英语单词 #搞笑" and Rosie finding it funny.
Enri: Got any funny meme for me this time?
Rosie: *chuckles* Oh, you know it! I've got a whole arsenal of meme magic up my sleeve. :woman-tipping-hand: But let me ask you this – have you seen the latest #Chrishere meme? :rolling_on_the_floor_laughing: It's a doozy! *winks* Want to see it? :tada:
'''


# %%
### Semantic Search
# You may construct the whole Memory Search database (with all friends' chat history)
c = Chat(device='mps')
c.vectorize()

# This allows you to do memory search freely with multiple friends
queries = ["good restaurants"]
friends = ["Eddie", "Andrew"]

contexts = {friend_name: c.semantic_search(queries, friend_name=friend_name) for friend_name in friends}
for (friend_name, context) in contexts.items():
    print(f"friend_name:{friend_name}")
    print(context)
    print()

# %%
### Freind Replica Chat Session
# 
model = ChatGLM()

chat_config = ChatConfig(
    my_name="Rosie",
    friend_name="王",
    language="chinese",
)
chat_with_friend = Chat(device='cpu', chat_config=chat_config)
m = LanguageModelwithRecollection(model, chat_with_friend, debug=True)

q = ''
current_chat = []
while q:
    q = input("")
    a = m(q, '\n'.join(current_chat))
    current_chat.append(chat_config.friend_name + ': ' + q)
    current_chat.append(chat_config.my_name + ': ' + a)
    