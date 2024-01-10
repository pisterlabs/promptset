from django.http import HttpResponse, JsonResponse
from mv_backend.lib.database import Database
from mv_backend.lib.common import CommonChatOpenAI
from mv_backend.settings import OPENAI_API_KEY
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage
)
import json, openai
from datetime import datetime
from bson.objectid import ObjectId

db = Database()

openai.api_key = OPENAI_API_KEY
import os
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

chat = CommonChatOpenAI()

# every prompts for this code

# CEFR criteria
CEFR = """pre-A1
Listening & Speaking Reading & Writing
CAN understand letters of the English alphabet when heard
CAN understand some simple spoken instructions given in short,
simple phrases
CAN understand some simple spoken questions about self - such as
name, age, favourite things or daily routine
CAN understand some very simple spoken descriptions of people
- such as name, gender, age, mood, appearance or what they
are doing
CAN understand some very simple spoken descriptions of everyday
objects - such as how many, colour, size or location
CAN understand very some short conversations that use familiar
questions and answers
CAN name some familiar people or things - such as family, animals,
and school or household objects
CAN give very basic descriptions of some objects and animals -
such as how many, colour, size or location
CAN respond to very simple questions with single words or a
‘yes/no’ response
CAN read and understand some simple sentences,
including questions
CAN follow some very short stories written in very simple language
CAN write the letters of the English alphabet
CAN write name using the English alphabet
CAN copy words, phrases and short sentences
CAN spell some very simple words correctly

A1
Listening & Speaking Reading & Writing
CAN understand very simple spoken dialogues about familiar topics
with the help of pictures
CAN understand very simple spoken descriptions about people
and objects
CAN express agreement or disagreement with someone using
short, simple phrases
CAN respond to questions on familiar topics with simple phrases
and sentences
CAN give simple descriptions of objects, pictures and actions
CAN tell a very simple story with the help of pictures
CAN ask someone how they are and ask simple questions about
habits and preferences
CAN understand some simple signs and notices
CAN read and understand some short factual texts with the help
of pictures
CAN read and understand some short, simple stories about familiar
topics with the help of pictures
CAN write short, simple phrases and sentences about pictures and
familiar topics
CAN write simple sentences giving personal details
CAN write short, simple sentences about likes and dislikes
8 CAMBRIDGE ENGLISH: YOUNG LEARNERS HANDBOOK FOR TEACHERS

A2
Listening & Speaking Reading & Writing
CAN understand instructions given in more than one sentence
CAN understand simple spoken descriptions of objects, people
and events
CAN understand simple conversations on everyday topics
CAN ask basic questions about everyday topics
CAN tell short, simple stories using pictures or own ideas
CAN give simple descriptions of objects, pictures and actions
CAN talk briefly about activities done in the past
CAN understand simple written descriptions of objects, people
and events
CAN understand simple, short stories containing narrative tenses
CAN read and understand short texts, even if some words
are unknown
CAN link phrases or sentences with connectors like ‘and’, ‘because’
and ‘then’
CAN write simple descriptions of objects, pictures and actions
CAN write a short, simple story using pictures or own ideas

RANGE	ACCURACY	FLUENCY	INTERACTION	COHERENCE
C2	Shows great flexibility reformulating ideas in differing linguistic forms to convey finer shades of meaning precisely, to give emphasis, to differentiate and to eliminate ambiguity. Also has a good command of idiomatic expressions and colloquialisms	Maintains consistent grammatical control of complex language, even while attention is otherwise engaged (e.g. in forward planning, in monitoring others' reactions).	Can express him/herself spontaneously at length with a natural colloquial flow, avoiding or backtracking around any difficulty so smoothly that the interlocutor is hardly aware of it.	Can interact with ease and skill, picking up and using non-verbal and intonational cues apparently effortlessly. Can interweave his/her contribution into the joint discourse with fully natural turntaking, referencing, allusion making etc.	Can create coherent and cohesive discourse making full and appropriate use of a variety of organisational patterns and a wide range of connectors and other cohesive devices.
C1	Has a good command of a broad range of language allowing him/her to select a formulation to express him/ herself clearly in an appropriate style on a wide range of general, academic, professional or leisure topics without having to restrict what he/she wants to say.	Consistently maintains a high degree of grammatical accuracy; errors are rare, difficult to spot and generally corrected when they do occur.	Can express him/herself fluently and spontaneously, almost effortlessly. Only a conceptually difficult subject can hinder a natural, smooth flow of language.	Can select a suitable phrase from a readily available range of discourse functions to preface his remarks in order to get or to keep the floor and to relate his/her own contributions skilfully to those of other speakers.	Can produce clear, smoothly-flowing, well-structured speech, showing controlled use of organisational patterns, connectors and cohesive devices.
B2	Has a sufficient range of language to be able to give clear descriptions, express viewpoints on most general topics, without much conspicuous searching for words, using some complex sentence forms to do so.	Shows a relatively high degree of grammatical control. Does not make errors which cause misunderstanding, and can correct most of his/her mistakes.	Can produce stretches of language with a fairly even tempo; although he/she can be hesitant as he or she searches for patterns and expressions, there are few noticeably long pauses.	Can initiate discourse, take his/her turn when appropriate and end conversation when he / she needs to, though he /she may not always do this elegantly.  Can help the discussion along on familiar ground confirming comprehension, inviting others in, etc.	Can use a limited number of cohesive devices to link his/her utterances into clear, coherent discourse, though there may be some "jumpiness" in a long contribution.
B1	Has enough language to get by, with sufficient vocabulary to express him/herself with some hesitation and circum-locutions on topics such as family, hobbies and interests, work, travel, and current events.	Uses reasonably accurately a repertoire of frequently used "routines" and patterns associated with more predictable situations.	Can keep going comprehensibly, even though pausing for grammatical and lexical planning and repair is very evident, especially in longer stretches of free production.	Can initiate, maintain and close simple face-to-face conversation on topics that are familiar or of personal interest. Can repeat back part of what someone has said to confirm mutual understanding.	Can link a series of shorter, discrete simple elements into a connected, linear sequence of points.
A2	Uses basic sentence patterns with memorised phrases, groups of a few words and formulae in order to communicate limited information in simple everyday situations.	Uses some simple structures correctly, but still systematically makes basic mistakes.	Can make him/herself understood in very short utterances, even though pauses, false starts and reformulation are very evident.	Can answer questions and respond to simple statements. Can indicate when he/she is following but is rarely able to understand enough to keep conversation going of his/her own accord.	Can link groups of words with simple connectors like "and, "but" and "because".
A1	Has a very basic repertoire of words and simple phrases related to personal details and particular concrete situations.	Shows only limited control of a few simple grammatical structures and sentence patterns in a memorised repertoire.	Can manage very short, isolated, mainly pre-packaged utterances, with much pausing to search for expressions, to articulate less familiar words, and to repair communication.	Can ask and answer questions about personal details. Can interact in a simple way but communication is totally dependent on repetition, rephrasing and repair.	Can link words or groups of words with very basic linear connectors like "and" or "then".
"""

# prompt: generate the CEFR score
#
# input:
# {CEFR} -- cuser's name
# {name} -- the number of queries
# {query} -- converstation (user -> npc)
cefr_template = """
I want you to act as an English teacher and professional English level assessor based on CEFR. First, learn the following CEFR guidelines for assessing English fluency. Here are the guidelines: 
{CEFR}
I want you to use the guidelines to assess {name}'s English. Find an accurate level assessment. Then answer that is *one word* must be in "pre-A1", "A1", "A2", "B1", "B2", "C1", "C2", "Idk". This is what I want you to assess: 
{query}
"""

cefr_prompt = PromptTemplate(
    input_variables=["CEFR", "name", "query"],
    template=cefr_template,
)

generate_cefr = LLMChain(
    llm=chat,
    prompt=cefr_prompt
)
# cefr_gpt:
# generate the CEFR score
def cefr_gpt(user, chat_data_list):
    data_num = 0
    all_chat_data_string = ""
    for chat_data in reversed(chat_data_list):
        data_num += 1
        if data_num > 30:
            break
        all_chat_data_string += chat_data + "\n"
    
    # generate the CEFR score
    cur_cefr = generate_cefr.run(CEFR = CEFR, name = user, query = all_chat_data_string)
    cefr = Database.get_recent_documents(db, user, "CEFR", 1)
    
    cefr_string = "Idk"
    now_cefr = ""
    now_cefr = cur_cefr
    for i in cefr:
        cefr_string = i["cefr"]

    if cur_cefr == "Idk":
        cur_cefr = cefr_string
        now_cefr = "Idk"
    
    cefr_data = Database.get_all_documents(db, user, "CEFR")
    node = 0
    data_num = 0

    for i in cefr_data:
        data_num += 1
    
    if data_num != 0:
        node = i["node"] + 1
    
    datetimeStr = datetime.now().strftime("%Y-%m-%d")
    
    document_user = {"_id":ObjectId(),"node":node,"timestamp":datetimeStr,"cefr":cur_cefr}

    print(Database.set_document(db, user, "CEFR_GPT", document_user))

    return now_cefr