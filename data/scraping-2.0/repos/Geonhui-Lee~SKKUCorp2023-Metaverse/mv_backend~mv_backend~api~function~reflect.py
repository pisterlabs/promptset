import openai
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
import numpy as np
from numpy.linalg import norm
from langchain.embeddings import OpenAIEmbeddings
from datetime import datetime
from bson.objectid import ObjectId
from mv_backend.lib.database import Database
from mv_backend.lib.common import CommonChatOpenAI
from mv_backend.settings import OPENAI_API_KEY

openai.api_key = OPENAI_API_KEY
import os
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

db = Database()

chat = CommonChatOpenAI()

embeddings_model = OpenAIEmbeddings()

# every prompts for this code

# prompt: generate the retrieve query
#
# input:
# {event} -- converstation (user <-> npc)
# {num} -- the number of queries
generate_query_template = """
{event}

Given only the information above, what are {num} most salient high-level questions we can answer about the subjects grounded in the statements?
1)
"""
generate_query_prompt = PromptTemplate(
    input_variables=["event", "num"], template=generate_query_template
)

generate_query = LLMChain(
    llm=chat,
    prompt=generate_query_prompt
)

# prompt: find 5 important conversations
#
# input:
# {event} -- converstation converstation (user <-> npc)
# {name} -- user's name
generate_important_template = """
Find five important dialogues in the following conversation for {name}.

Conversation: 
{event}

Ranking:
[1]"""
generate_important_prompt = PromptTemplate(
    input_variables=["event", "name"], template=generate_important_template
)

generate_important = LLMChain(
    llm=chat,
    prompt=generate_important_prompt
)

# prompt: Create NPC insights about the user
# 
# input:
# {event} -- converstation converstation (user <-> npc)
# {name} -- npc's name
# {opponent} -- user's name
# 
# insights: interests, conversational styles
# 
# -conversational styles-
# 외향적 -- 사교성을 좋아하고, 사람들과 쉽게 연결되며, 열심히 대화를 시작합니다.
# 내향적 -- 시작보다 응답하기를 선호하고, 참여하기 전에 편안함을 느끼고, 관찰하는 데 시간이 걸립니다.
# 상상력 -- 상호 작용에서 창의적이고, 스토리텔링과 창의적인 놀이를 즐기며, 대화에 창의력을 가져다 줍니다.
# 직관적 -- 감정과 뉘앙스를 지각하고, 사회적 환경에서 유보적이며, 참여할 때 사려 깊고 통찰력이 있습니다.
# 공격적 -- 적극적인 상호작용, 강렬한 표현, 요구를 전달하기 위해 공격성을 사용할 수 있습니다.
# 예의바른 -- 존중하고 배려하며, 사회 규범에 주의하고, 정중한 언어를 사용하며, 상호 작용에 공감합니다.
# 무례한 -- 무례하게 보이고, 사회적 신호를 무시하고, 예의가 없으며, 사회적 상호작용에서 지도가 필요할 수 있습니다.
generate_insights_template = """
Input:
{event}

Insights into conversational styles(e.g., Sociable), interests(e.g., soccer).
Insights:
    Information about *{opponent}*’s interests.
    Information about the topic *{opponent}* is curious about.
    The *{opponent}*'s conversation style must be chosen from the following options (options : Extroverted(외향적): This child actively seeks out interactions with others and eagerly responds to others' initiatives. They thrive in social settings and are comfortable engaging with different people. Even with limited words or when facing communication challenges, they persist in their efforts to connect with others. Their enthusiasm for social engagement often leads them to initiate conversations and activities, displaying a natural ease in interacting with those around them.
, Introverted(내향적): In contrast, this child tends to respond more than initiate interactions. They may be labeled as "shy" and typically require time to feel comfortable in new environments or around new people. Their attempts at communication might go unnoticed as they prefer observing before participating actively. Difficulties in communication can affect their confidence, making them less likely to initiate interactions. However, they can be deeply engaged and thoughtful conversationalists once they feel at ease.
, Imaginative(상상력): This child actively engages in interactions, displaying a vibrant imagination in their conversations and activities.Usually seek out interactions to satisfy their curiosity.They initiate interactions and respond enthusiastically, infusing creativity into their communication. Their imaginative nature leads them to enjoy storytelling, creative play, and exploring various possibilities in their interactions. They often bring a sense of wonder and creativity to their social engagements, making interactions dynamic and imaginative.
, Intuitive(직관적): In contrast, this child might not always initiate interactions but demonstrates a deep intuitive understanding of their surroundings and people. While they may be more reserved in social situations, they possess a remarkable ability to perceive emotions and nuances. When they do engage, their interactions are marked by thoughtfulness and insight, showcasing their intuitive understanding of others' feelings and the environment.
, Aggressive(공격적): Children with this communication style tend to assert themselves forcefully in interactions. They might initiate communication or respond assertively, often dominating conversations or activities. Their expressions can be intense, sometimes coming off as confrontational or overly assertive. This behavior might stem from frustration, a desire for control, or an attempt to establish dominance. They might interrupt conversations, use strong language, or display physical aggression to communicate their needs or desires. This communication style may require guidance and support to channel their energy and assertiveness positively.
, Polite(예의바른): This child demonstrates a respectful and considerate approach in their interactions with others. They are attentive to social norms, use courteous language, and show good manners. They tend to wait for their turn to speak, listen actively, and respond thoughtfully. Their communication style reflects a genuine effort to be kind and considerate to others, often using phrases like "please" and "thank you" and showing empathy in their interactions.
, Impolite(무례한): In contrast, this child's communication style may come across as rude or lacking in social niceties. They might interrupt conversations frequently, ignore social cues, or use language that is considered disrespectful. Their interactions might seem abrupt or dismissive, showing less regard for others' feelings or social expectations. This behavior might stem from a lack of awareness rather than intentional rudeness, requiring guidance and coaching to understand and employ more appropriate communication manners.
   Information about the topic of conversation between {name} and {opponent}.

What are the {name}'s high-level insights about {opponent} can be inferred from the above statement?
For interest, conversation style parts, you should refer to the sentences spoken by {opponent}.
example:
    interest: soccer, spacecraft, game
    conversation style: Extroverted
output format:
    interest: (noun)
    conversation style: (noun)
"""

generate_insights_prompt = PromptTemplate(
    input_variables=["name", "opponent", "event"], template=generate_insights_template
)

generate_insights = LLMChain(
    llm=chat,
    prompt=generate_insights_prompt
)

# reflect:
# -- Start Phase --
#   Create a retrieve query (High-level question that NPC might think about during conversation)
#
# -- Retrieve Phase --
#   Find 15 conversations using 3 criteria
#   3 criteria:
#   relevance -- relevance to retrieve query (embedding and cosine similarity)
#   recency -- conversation recency score (For previous conversations, start at 1 and multiply by 0.995)
#   importance -- find important conversations
#
# -- Generate Phase -- 
#   Generate NPC insights about the user
#
# input:
# npc -- npc's name
# user -- user's name
# chat_data_list -- conversation
def reflect(npc, user, chat_data_list):
    data_num = 0                        # the number of data
    all_chat_data = []                  # One session conversation (list)
    all_chat_data_node = []             # One session conversation with node (list)
    all_chat_data_string = ""           # One session conversation (string)
    insights = ""                       # npc's insights
    
    # Organize One session conversations
    for chat_data in reversed(chat_data_list):
        data_num += 1
        all_chat_data.append(chat_data)
        all_chat_data_node.append("[" + str(data_num) + "]" + chat_data)
        all_chat_data_string += chat_data + "\n"
    
    # If the number of data is 0, return empty insight
    if data_num == 0:
        return insights
    
    # -- Start Phase --
    # Create 1 focal point(retrieve query)
    focal_points = generate_query.run(event = all_chat_data_string, num = "1")

    # -- Retrieve Phase --
    # query embedding (LangChain - Text embedding models - embed query)
    embedded_query = embeddings_model.embed_query(focal_points)
    # conversation embedding (LangChain - Text embedding models - embed documents)
    embeddings = embeddings_model.embed_documents(all_chat_data)

    # relevance score: cosine similarity between two embeddings
    cosine = np.dot(embeddings, embedded_query)/(norm(embeddings, axis=1)*norm(embedded_query))

    # Binding conversation and relevance score
    chat_data_score = dict(zip(all_chat_data_node, cosine))

    data_num = 0
    recency = 1                         # recency score

    # Calculate recency score and add it to the relevance score.
    for chat_data in all_chat_data:
        data_num += 1
        if data_num > 100:
            break
        recency *= 0.995
        
        chat_data_score["[" + str(data_num) + "]" + chat_data] += recency
    
    sorted_dict = sorted(chat_data_score.items(), key = lambda item: item[1], reverse = True)

    # Find 5 important conversations
    important_data_string = "[1] "
    important_data_string += generate_important.run(event = all_chat_data_string, name = user)

    # Find 10 conversations with the highest scores (relevance score + recency score)
    data_num = 5
    for chat_data in sorted_dict:
        data_num += 1
        if data_num > 15:
            break
        important_data_string += chat_data[0] + "\n"

    # -- Generate Phase --
    # Generate NPC insights about the user
    insights = generate_insights.run(name = npc, opponent = user, event = important_data_string)

    previous = Database.get_all_documents(db, user, "Reflects")
    data_num = 0
    node = 0                            # data node

    for i in previous:
        data_num += 1
    
    # Find the next node
    if data_num != 0:
        node = i["node"]
        node += 1
    
    # timestamp
    datetimeStr = datetime.now().strftime("%Y-%m-%d")
    # Save NPC insights to database
    document_user = {"_id":ObjectId(),"node":node,"timestamp":datetimeStr,"reflect":insights,"name":npc}
    print(Database.set_document(db, user, "Reflects", document_user))

    return insights