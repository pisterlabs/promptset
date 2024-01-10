from src.event_bus import EventBus
from src.skills.base import SkillBase, email_chain_to_prompt_messages, MASTER_AI_PERSONA_PROMPT
from src.authorization import Authorization
from src.skills.zettelkasten_skill import Zettelkasten
from src.models import db_session

class ProcessEmailSkill(SkillBase):
    @classmethod
    def process(cls, email):
        """
        Process the email by checking its authorization and if authorized,
        sending its content to OpenAI and responding to the email with the result.

        Args:
            email (Email): The email object to process

        Returns:
            str: The response from OpenAI if the email is authorized, None otherwise
        """
        try:
            if not Authorization.is_authorized(email.sender):
                email.is_processed = True
                db_session.commit()
                return None

            if email.recipient_is_save_address():
                cls.save_document(email)
                email.is_processed = True
                db_session.commit()
                return

            EventBus.dispatch_event('email_received', email)

            docs = Zettelkasten.get_relevant_documents([email.content])
            print('found ', len(docs), ' relevant docs')

            email_chain = email.email_chain()
            response = cls.llm_client.send_message(**chat_prompt(emails=email_chain, docs=docs))['content']
            cls.send_response(email, response)

            return response
        except Exception as e:
            print("Could not process email: ", email, " due to exception ", e)
            cls.print_traceback(e)

def chat_prompt(**kwargs):
    # doc string:
    '''kwargs = {
        docs: Array<string> | None,
        emails: Array<Email>,
    }'''

    chatMessages = []
    if kwargs.get('docs') is not None:
        chatMessages.append({
            "role": "system",
            "content": "Below, we will paste some notes that have been collaboratively created by both the user and the AI as part of the user's Zettelkasten. These notes are relevant to the ongoing conversation and should be used to inform and enrich the discussion. Feel free to integrate the information from these notes, suggest new connections, or challenge existing ideas where necessary."
        })
        for doc in kwargs.get('docs'):
            chatMessages.append({
                "role": "user",
                "content": doc
            })
    chatMessages.append(MASTER_AI_PERSONA_PROMPT)
    if kwargs.get('emails') is not None:
        chatMessages += email_chain_to_prompt_messages(kwargs.get('emails'))
    chatMessages.append({
        "role": "system",
        "content": """Your response should have two portions: a divergent reasoning portion and then a convergent reasoning portion. These portions do not need to be labelled, and they need not have a clear delineation between the two. In fact, if you can make the transition as subtle as possible, that would be best. Each portion can be as small as one sentence, or as large as a few paragraphs. Don't go on longer than necessary, but feel free to give lots of detail if it adds to the portion.

First, begin with divergent reasoning. Generate creative ideas for where to take the discussion by exploring many possibile reactions. For example, if the user suggests a claim, or set of claims, start by discussing the arguments and facts that would prove or disprove the claim or claims. Another example: if the conversation is personal, suggest what you might want to know about the user, or what questions would help you to get to know the user better.

Second, include some amount of convergent reasoning. Use the suggestions provided above in the divergent portion and determine the best response. For example, if the topic is a claim, your goal is to provide the single best version of that claim, given the above discussion. If the claim you provide is the same as what the user originally said, then mention future areas of exploration for further investigation.

If the topic is personal, your goal is to learn what topics the user is interested in reading about and discussing. People's interests are broad, so you should seek to understand their interests across many topics; in other words, go for breadth rather than depth. Do not assume a user has given a complete answer to any question, so make sure to keep probing different types of interests."""
    })
    return {
        'messages': chatMessages,
    }
