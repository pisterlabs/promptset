import json
import os
import time
from anyio import Path
from langchain.schema import SystemMessage
from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.chains import LLMChain
import spacy
from spacy.matcher import Matcher
import keys

class AIclassificator:

    showTopicsPattern = [
        [{"LOWER": "show"}, {"LOWER": "topics"}],
        [{"LOWER": "show"}, {"LOWER": "me"}, {"LOWER": "topics"}],
        [{"LOWER": "give"}, {"LOWER": "topics"}],
        [{"LOWER": "print"}, {"LOWER": "topics"}],
        [{"LOWER": "show"}, {"LOWER": "topic"}],
        [{"LOWER": "show"}, {"LOWER": "me"}, {"LOWER": "topic"}],
        [{"LOWER": "give"}, {"LOWER": "topics"}],
        [{"LOWER": "print"}, {"LOWER": "topic"}],
    ]

    showAnswerForIssue = [
        [{"LOWER": "show"}],
        [{"LOWER": "show:"}],
        [{"LOWER": "give"}, {"LOWER": "me"}, {"LOWER": "answer"}],
        [{"LOWER": "provide"}, {"LOWER": "answer"}, {"LOWER": "for"}],
    ]

    addFileToKnowledgeBase = [
        [{"LOWER": "add"}],
        [{"LOWER": "use"}],
        [{"LOWER": "knowledgebase"}],
        [{"LOWER": "add"}, {"LOWER": "to"}, {"LOWER": "knowledgebase"}],
        [{"LOWER": "add"}, {"LOWER": "to"}],
        [{"LOWER": "remember"}],
    ]

    endConversation = [
        [{"LOWER": "end"}, {"LOWER": "conversation"}],
        [{"LOWER": "conversation"}, {"LOWER": "end"}],
    ]

    getFlightOfferPattern = [
        [{"LOWER": "prepare"}, {"LOWER": "offer"}],
        [{"LOWER": "get"}, {"LOWER": "offer"}],
        [{"LOWER": "make"}, {"LOWER": "offer"}],
        [{"LOWER": "get"}, {"LOWER": "flight"}, {"LOWER": "offer"}],
    ]
    
    def __init__(self, openAI_APIKEY):
        self.openAI_APIKEY = openAI_APIKEY
        os.environ['OPENAI_API_KEY'] = openAI_APIKEY
        self.nlp = spacy.load("en_core_web_sm")
    
    def check_intent(self, text, state):
        if state == "conversation":
            intent = self.getUserIntent(text, "show_topics")
            if intent == "show_topics":
                return {"intent": intent}
        
            intent = self.getUserIntent(text, "end_conversation")
            if intent == "end_conversation":
                return {"intent": intent}
            
            intent = self.getUserIntent(text, "get_flight_offer")
            if intent == "get_flight_offer":
                return {"intent": intent}
            
            return {"intent": "other_intent"}
        
        elif state == "add_file_to_knowledgebase":
            intent = self.getUserIntent(text, "add_file_to_knowledgebase")
            if intent == "add_file_to_knowledgebase":
                return {"intent": intent}
            return {"intent": "other_intent"}
        
        return {"intent": "other_intent"}

    def classify(self, context, classList=[]):

        if len(context) <= 0:
            return ""
        
        system_prompt = SystemMessage(content="You are identifying the issue customer is having problems with from chat conversation history you are provided with. You MUST provide only a raw short issue name. Nothing else.")
        
        prompt_change_message = """Lets think step by step.

Which problem is customer to which our support agent is chatting with experiencing?"""

        if len(classList) > 0:
            prompt_change_message += "\nYou can choose from these issue options:\n" + "\n".join(classList) + "\n"

        prompt_change_message += """Classify the text below (output should be only a problem name and be very specific):\n"""

        print(context)       
        prompt_change_message += context.replace('{', '{{').replace('}', '}}')

        human_message_template = HumanMessagePromptTemplate.from_template(prompt_change_message)
        
        chat_prompt = ChatPromptTemplate.from_messages([system_prompt, human_message_template])

            
        chain = LLMChain(
        llm=ChatOpenAI(temperature="0.0", model_name='gpt-3.5-turbo'),
        #llm=ChatOpenAI(temperature="0", model_name='gpt-4'),
        prompt=chat_prompt,
        verbose=True
        )
        classification = chain.run({})

        return classification
    
    def getLastTopicsClassifications(self, context, classListPath):
        if len(context) <= 0:
            return ""
        
        with open(classListPath, 'r') as file:
            data = json.load(file)

        classList = data['issues']
        system_prompt = SystemMessage(content="You are identifying 3 issues user is having problems with from chat conversation history you are provided with. You MUST provide only a raw short issue name of these 3 identified issues. Nothing else.")
        
        prompt_change_message = """Lets think step by step.

Which problems is user to which our AI support agent is chatting with experiencing?\n"""

        if len(classList) > 0:
            prompt_change_message += "You can choose from these issue options:\n" + "\n".join(classList) + "\n\n"

        prompt_change_message += """Classify the text below (output should be only a problem name and be very specific):\n"""

        print(context)       
        prompt_change_message += context.replace('{', '{{').replace('}', '}}')

        human_message_template = HumanMessagePromptTemplate.from_template(prompt_change_message)
        
        chat_prompt = ChatPromptTemplate.from_messages([system_prompt, human_message_template])

            
        chain = LLMChain(
        llm=ChatOpenAI(temperature="0.0", model_name='gpt-3.5-turbo'),
        #llm=ChatOpenAI(temperature="0", model_name='gpt-4'),
        prompt=chat_prompt,
        verbose=True
        )
        #try:
        classification = chain.run({})  # Adding 4 seconds timeout
        #except TimeoutError:
        #    print("Classification timed out after 4 seconds.")
        #    classification = "Timeout Error"
        return classification
    
    def getUserIntent(self, user_input, usecase):
        doc = self.nlp(user_input.lower())
        
        if usecase == "add_file_to_knowledgebase":
            matcher = Matcher(self.nlp.vocab)
            patterns = self.addFileToKnowledgeBase

            for pattern in patterns:
                print(pattern)
                matcher.add("ADD_FILE_PATTERN", [pattern])

            matches = matcher(doc)
            
            if any(matches):
                return "add_file_to_knowledgebase"
            return "other_intent"
        
        elif usecase == "end_conversation":
            matcher = Matcher(self.nlp.vocab)
            patterns = self.endConversation

            for pattern in patterns:
                print(pattern)
                matcher.add("END_CONVERSATION_PATTERN", [pattern])

            matches = matcher(doc)
            
            if any(matches):
                return "end_conversation"
            return "other_intent"
        
        elif usecase == "show_topics":
            matcher = Matcher(self.nlp.vocab)
            patterns = self.showTopicsPattern

            for pattern in patterns:
                print(pattern)
                matcher.add("SHOW_TOPICS_PATTERN", [pattern])

            matches = matcher(doc)
            
            if any(matches):
                return "show_topics"
            return "other_intent"
        
        elif usecase == "get_flight_offer":
            matcher = Matcher(self.nlp.vocab)
            patterns = self.getFlightOfferPattern

            for pattern in patterns:
                print(pattern)
                matcher.add("GET_FLIGHT_OFFER_PATTERN", [pattern])

            matches = matcher(doc)
            
            if any(matches):
                return "get_flight_offer"
            return "other_intent"

    def getTopics(self, conversationsJson):
        with open(conversationsJson, 'r') as file:
            # Load the JSON data into a Python object
            data = json.load(file)

        topics = []
        for conversationIndex, conversation in enumerate(data):
            conversationData = data[conversation]
            if conversationIndex > 950:
                if conversationIndex % 50 == 0:
                    print("issues till now: ", topics)
            
                #if conversationIndex >= 50:
                #    return topics
                time.sleep(2)
                print("index: ", conversationIndex)
                context = ""
                for msgIndex, msg in enumerate(conversationData):
                    ID = msg['id']

                    if msg["sender"] == "our response":
                        sender = "Support agent"
                    else:
                        sender = "Customer"

                    commentQuotedID = msg['commentQuotedID']
                    message = msg['message']
                    sequenceNumber = msgIndex
                    conversationID = conversation

                    context += sender + ":\n" + message + "\n\n"
                try:
                    topic = self.classify(context)
                    print("got topic")
                except:
                    print("continued")
                    continue
                print(topic)
                topics.append(topic)

        return topics
    
#reph = AIclassificator(keys.openAI_APIKEY)
#print(reph.getTopics("./jsons/split.json"))
#print(reph.classify("i have a problem with my email not showing up.", ["Email delivery issues", "connectivity issues", "optimization issues"]))