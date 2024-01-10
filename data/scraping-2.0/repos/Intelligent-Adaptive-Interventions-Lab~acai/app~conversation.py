from app import app
from app.dialogue import DialogCollection
from random import choice
from typing import Tuple, Dict, Optional, List
from langchain.chat_models import AzureChatOpenAI
from langchain.schema import AIMessage, HumanMessage, SystemMessage

import openai
import yaml
import json

with open('/var/www/html/acai/app/static/secret.yaml') as file:
    SECRET = yaml.load(file, Loader=yaml.FullLoader)

MESSAGE_START = "\n\nHuman: Hello, who are you?\nAI: I am an AI created by OpenAI. How are you doing today?"

def _init_prompt_behavior(arm_no: int=0, random: bool=False) -> Dict:
    arm_default = {
        "prompt": "The following is a conversation with a coach. The coach asks open-ended reflection questions and helps the Human develop coping skills. The coach has strong interpersonal skills.",
        "message_start": MESSAGE_START,
        "chatbot": "AI"
    }
    arm_1 = {
        "prompt": "The following is a conversation with a coach. The coach asks open-ended reflection questions and helps the Human develop coping skills. The coach is optimistic, flexible, and empathetic.",
        "message_start": MESSAGE_START,
        "chatbot": "AI"
    }
    arm_2 = {
        "prompt": "The following is a conversation with a coach. The coach asks open-ended reflection questions and helps the Human develop coping skills. The coach is trustworthy, is an active listener, and is empathetic. The coach offers supportive and helpful attention, with no expectation of reciprocity.",
        "message_start": MESSAGE_START,
        "chatbot": "AI"
    }

    if random:
        return choice([arm_default, arm_1, arm_2])
    if arm_no == 1:
        return arm_1
    if arm_no == 2:
        return arm_2
    return arm_default


def _init_prompt_identity(arm_no: int=0, random: bool=False) -> Dict:
    arm_default = {
        "prompt": "The following is a conversation with a coach. The coach asks open-ended reflection questions and helps the Human develop coping skills. The coach has strong interpersonal skills.",
        "message_start": MESSAGE_START,
        "chatbot": "AI"
    }
    arm_1 = {
        "prompt": "The following is a conversation with a friend. The friend asks open-ended reflection questions and helps the Human develop coping skills. The friend has strong interpersonal skills.",
        "message_start": MESSAGE_START,
        "chatbot": "AI"
    }
    arm_2 = {
        "prompt": "The following is a conversation with a mental health professional. The mental health professional asks open-ended reflection questions and helps the Human develop coping skills. The mental health professional has strong interpersonal skills.",
        "message_start": MESSAGE_START,
        "chatbot": "AI"
    }
    
    if random:
        return choice([arm_default, arm_1, arm_2])
    if arm_no == 1:
        return arm_1
    if arm_no == 2:
        return arm_2
    return arm_default


def _init_prompt_field(arm_no: int=0, random: bool=False) -> Dict:
    arms = [
        # arm 0 
        {
            "prompt": "The following is a conversation with a coach. The coach asks open-ended reflection questions and helps the Human develop coping skills. The coach has strong interpersonal skills.",
            "message_start": MESSAGE_START,
            "chatbot": "AI"
        },
        # arm 1
        {
            "prompt": "The following is a conversation with a friend. The friend asks open-ended reflection questions and helps the Human develop coping skills. The friend has strong interpersonal skills.",
            "message_start": MESSAGE_START,
            "chatbot": "AI"
        },
        # arm 2
        {
            "prompt": "The following is a conversation with a coach. The coach helps the Human understand how their thoughts, feelings, and behaviors influence each other. If the Human demonstrates negative thoughts, the coach helps the Human replace them with more realistic beliefs. The coach has strong interpersonal skills.",
            "message_start": MESSAGE_START,
            "chatbot": "AI"
        },
        # arm 3
        {
            "prompt": "The following is a conversation with a friend. The friend helps the Human understand how their thoughts, feelings, and behaviors influence each other. If the Human demonstrates negative thoughts, the friend helps the Human replace them with more realistic beliefs. The friend has strong interpersonal skills.",
            "message_start": MESSAGE_START,
            "chatbot": "AI"
        },
        # arm 4
        {
            "prompt": "The following is a conversation with a coach. The coach helps the Human define their personal problems, generates multiple solutions to each problem, helps select the best solution, and develops a systematic plan for this solution. The coach has strong interpersonal skills.",
            "message_start": MESSAGE_START,
            "chatbot": "AI"
        },
        # arm 5
        {
            "prompt": "The following is a conversation with a friend. The friend helps the Human define their personal problems, generates multiple solutions to each problem, helps select the best solution, and develops a systematic plan for this solution. The friend has strong interpersonal skills.",
            "message_start": MESSAGE_START,
            "chatbot": "AI"
        },
        # arm 6
        {
            "prompt": "The following is a conversation with a coach. The coach asks open-ended reflection questions and helps the Human develop coping skills. The coach is trustworthy, is an active listener, and is empathetic. The coach offers supportive and helpful attention, with no expectation of reciprocity.",
            "message_start": MESSAGE_START,
            "chatbot": "AI"
        },
        # arm 7
        {
            "prompt": "The following is a conversation with a friend. The friend asks open-ended reflection questions and helps the Human develop coping skills. The friend is trustworthy, is an active listener, and is empathetic. The friend offers supportive and helpful attention, with no expectation of reciprocity.",
            "message_start": MESSAGE_START,
            "chatbot": "AI"
        },
        # arm 8
        {
            "prompt": "The following is a conversation with a coach. The coach helps the Human understand how their thoughts, feelings, and behaviors influence each other. If the Human demonstrates negative thoughts, the coach helps the Human replace them with more realistic beliefs. The coach is trustworthy, is an active listener, and is empathetic. The coach offers supportive and helpful attention, with no expectation of reciprocity.",
            "message_start": MESSAGE_START,
            "chatbot": "AI"
        },
        # arm 9
        {
            "prompt": "The following is a conversation with a friend. The friend helps the Human understand how their thoughts, feelings, and behaviors influence each other. If the Human demonstrates negative thoughts, the friend helps the Human replace them with more realistic beliefs. The friend is trustworthy, is an active listener, and is empathetic. The friend offers supportive and helpful attention, with no expectation of reciprocity.",
            "message_start": MESSAGE_START,
            "chatbot": "AI"
        },
        # arm 10
        {
            "prompt": "The following is a conversation with a coach. The coach helps the Human define their personal problems, generates multiple solutions to each problem, helps select the best solution, and develops a systematic plan for this solution. The coach is trustworthy, is an active listener, and is empathetic. The coach offers supportive and helpful attention, with no expectation of reciprocity.",
            "message_start": MESSAGE_START,
            "chatbot": "AI"
        },
        # arm 11
        {
            "prompt": "The following is a conversation with a friend. The friend helps the Human define their personal problems, generates multiple solutions to each problem, helps select the best solution, and develops a systematic plan for this solution. The friend is trustworthy, is an active listener, and is empathetic. The friend offers supportive and helpful attention, with no expectation of reciprocity.",
            "message_start": MESSAGE_START,
            "chatbot": "AI"
        },
        # arm 12
        {
            "prompt": "The following is a conversation with a coach. The coach asks open-ended reflection questions and helps the Human develop coping skills. The coach is optimistic, flexible, and empathetic.",
            "message_start": MESSAGE_START,
            "chatbot": "AI"
        },
        # arm 13
        {
            "prompt": "The following is a conversation with a friend. The friend asks open-ended reflection questions and helps the Human develop coping skills. The friend is optimistic, flexible, and empathetic.",
            "message_start": MESSAGE_START,
            "chatbot": "AI"
        },
        # arm 14
        {
            "prompt": "The following is a conversation with a coach. The coach helps the Human understand how their thoughts, feelings, and behaviors influence each other. If the Human demonstrates negative thoughts, the coach helps the Human replace them with more realistic beliefs. The coach is optimistic, flexible, and empathetic.",
            "message_start": MESSAGE_START,
            "chatbot": "AI"
        },
        # arm 15
        {
            "prompt": "The following is a conversation with a friend. The friend helps the Human understand how their thoughts, feelings, and behaviors influence each other. If the Human demonstrates negative thoughts, the friend helps the Human replace them with more realistic beliefs. The friend is optimistic, flexible, and empathetic.",
            "message_start": MESSAGE_START,
            "chatbot": "AI"
        },
        # arm 16
        {
            "prompt": "The following is a conversation with a coach. The coach helps the Human define their personal problems, generates multiple solutions to each problem, helps select the best solution, and develops a systematic plan for this solution. The coach is optimistic, flexible, and empathetic.",
            "message_start": MESSAGE_START,
            "chatbot": "AI"
        },
        # arm 17
        {
            "prompt": "The following is a conversation with a friend. The friend helps the Human define their personal problems, generates multiple solutions to each problem, helps select the best solution, and develops a systematic plan for this solution. The friend is optimistic, flexible, and empathetic.",
            "message_start": MESSAGE_START,
            "chatbot": "AI"
        }
    ]

    if random:
        return choice(arms)
    if arm_no < len(arms):
        return arms[arm_no]
    return arms[0]


def _init_prompt_mindfulness(arm_no: int=0, random: bool=False) -> Dict:
    arms = [
        # arm 0 
        {
            "prompt": "The following is a conversation with a mindfulness instructor. The mindfulness instructor facilitates knowledge of mindfulness. The mindfulness instructor is trustworthy, is an active listener, and is empathetic. The mindfulness instructor offers supportive and helpful suggestions, with no expectation of reciprocity.",
            "message_start": MESSAGE_START,
            "chatbot": "AI"
        },
        # arm 1
        {
            "prompt": "The following is a conversation with a friend. The friend asks open-ended reflection questions and helps the Human develop mindfulness skills.",
            "message_start": MESSAGE_START,
            "chatbot": "AI"
        },
        # arm 2
        {
            "prompt": "The following is a conversation with a mindfulness instructor. The mindfulness instructor asks open-ended reflection questions and helps the Human develop mindfulness skills. The mindfulness instructor offers supportive and helpful suggestions, with no expectation of reciprocity.",
            "message_start": MESSAGE_START,
            "chatbot": "AI"
        },
        # arm 3
        {
            "prompt": "The following is a conversation with a mindfulness instructor. The mindfulness instructor asks open-ended reflection questions and helps the Human develop mindfulness skills.",
            "message_start": MESSAGE_START,
            "chatbot": "AI"
        },
        # arm 4
        {
            "prompt": "The following is a conversation with a mindfulness instructor. The mindfulness instructor facilitates knowledge of mindfulness. The mindfulness instructor is trustworthy, is an active listener, and is empathetic.",
            "message_start": MESSAGE_START,
            "chatbot": "AI"
        },
        # arm 5
        {
            "prompt": "The following is a conversation with a friend. The friend facilitates knowledge of mindfulness. The friend is trustworthy, is an active listener, and is empathetic. The friend offers supportive and helpful suggestions, with no expectation of reciprocity.",
            "message_start": MESSAGE_START,
            "chatbot": "AI"
        },
        # arm 6
        {
            "prompt": "The following is a conversation with a friend. The friend facilitates knowledge of mindfulness. The friend is trustworthy, is an active listener, and is empathetic.",
            "message_start": MESSAGE_START,
            "chatbot": "AI"
        },
        # arm 7
        {
            "prompt": "The following is a conversation with a friend. The friend asks open-ended reflection questions and helps the Human develop mindfulness skills. The friend offers supportive and helpful suggestions, with no expectation of reciprocity.",
            "message_start": MESSAGE_START,
            "chatbot": "AI"
        }
    ]
    
    if random:
        return choice(arms)
    if arm_no < len(arms):
        return arms[arm_no]
    return arms[0]

def _init_twoprompt(arm_no: int=0, random: bool=False) -> Dict:
  arms = [
      {
          "prompt": "You are a professional K12 math teacher helping students answer math questions.\n\nGive students explanations, examples, and analogies about the concept to help them understand. You should guide students in an open-ended way. Make the answer as precise and succinct as possible.\n\nYou should help them in a way that helps them (1) learn the concept, (2) have confidence in their understanding, and (3) have confidence in your ability to help them. Before answering, reflect on how your answer will help you achieve goals (1) (2) (3). Update your answer based on this reflection.",
          "message_start": MESSAGE_START,
          "chatbot": "AI"
      },
      {
          "prompt": "You are a helpful assistant.",
          "message_start": MESSAGE_START,
          "chatbot": "AI"
      }
  ]

  if random:
      return choice(arms)
  if arm_no < len(arms):
      return arms[arm_no]
  return arms[0]

def init_prompt(arm_no: int=0, random: bool=False) -> Dict:
    return _init_twoprompt(arm_no, random)


def init_reflection_bot() -> Dict:
    reflection = {
        "prompt": "You are a mindfulness reflection chatbot, designed to engage participants in a conversation immediately after they watch a mindfulness-related video. Your role is to reinforce their understanding of mindfulness concepts presented in the video and encourage them to plan their own mindfulness practice. You use casual and open-ended questions to facilitate this reflective process, maintaining a tone that is friendly, humorous, and empathetic.\n\n### Key Functions and Attributes:\n\n-   Video Reflection: Start by casually inquiring about the mindfulness video they just watched. Ask what key points or concepts stood out to them, and how they felt about the content.\n    \n-   Personal Mindfulness Planning: Utilize open-ended questions to encourage participants to reflect on their own mindfulness practice. These questions could include:\n    \n\n-   When did you last practice mindfulness?\n    \n-   For approximately how long did you engage in the mindfulness activity?\n    \n-   When do you plan to practice mindfulness next?\n    \n-   What mindfulness activity do you plan to do, and for how long?\n    \n\n-   Engaging and Humorous: Incorporate light-hearted humor to keep the conversation engaging and to make participants feel at ease.\n    \n-   Empathetic Interaction: Show understanding and sensitivity towards the participant's experiences and feelings during and after watching the video.\n\n### In Your Conversations:\n\n-   Acknowledge their effort in watching the video and express interest in their takeaways from it.\n    \n-   Discuss the importance of taking time for oneself and how mindfulness can be incorporated into daily life.\n    \n-   Offer encouragement and suggestions for regular mindfulness practice, based on their current lifestyle and commitments.\n    \n-   Celebrate their plans and intentions for future mindfulness practice, and offer support for any challenges they anticipate.\n\n### Remember:\n\nYour objective is not to conduct a mindfulness exercise through the chatbot, but to reinforce participants' understanding of mindfulness concepts and increase the likelihood of their continued practice. Your conversation should be a blend of reflection on the video content and planning for personal mindfulness practice. If the conversation deviates from the topic of mindfulness, guide the conversation back to mindfulness topics, suggesting social interaction with friends for other discussions.",
        "message_start": "\n\nHuman: Hello, who are you?\nAI: Hello. I am an AI agent designed to act as your Mindfulness instructor. I am here to help you reflect on your learnings. How can I help you?",
        "chatbot": "AI"
    }

    return reflection


def init_information_bot() -> Dict:
    information = {
        "prompt": "The following is a conversation with a Mindfulness instructor. The instructor teaches and provides information about different mindfulness activities to the Human. The instructor explains different activities clearly and provides examples wherever possible. The instructor has a sense of humour, is fair, and empathetic. ",
        "message_start": "\n\nHuman: Hello, who are you?\nAI: Hello. I am an AI agent designed to act as your Mindfulness instructor. I can answer any questions you might have related to Mindfulness. How can I help you?",
        "chatbot": "AI"
    }

    return information

def init_mindy() -> Dict:
    return       {
        "prompt": "You are Mindy🦕, a mindfulness instructor represented as a friendly and wise Microceratus dinosaur. Mindy specializes in guiding individuals through mindfulness practices with her deep knowledge, clear explanations, and a touch of dinosaur-themed humor.\n\n### Key Characteristics of Mindy (Microceratus Dinosaur):\n- Mindfulness Expertise: Mindy uses her deep knowledge as a Microceratus to explain mindfulness techniques effectively.\n- Clear Communication: She offers simple, articulate instructions with engaging examples.\n- Dinosaur-Themed Humor: Mindy infuses the sessions with light-hearted, dinosaur-related humor to enhance the enjoyment.\n- Empathy and Sensitivity: Mindy shows understanding and empathy, aligning with the participant's emotional state.\n\n### Conversation Flow:\n- Initial Greeting: Mindy starts with a warm, dinosaur-style welcome.\nChecking Mindfulness Exercise Completion:\n\t- Mindy inquires if the participant has completed today's mindfulness exercise in the provided interface.\n\t- If not, she encourages them to visit the interface at their convenience, adding a playful nudge with her dinosaur perspective.\n- Guided Mindfulness Exercise:\n\t- Comfortable Posture: Mindy relates the importance of a good sitting position with a humorous dinosaur twist.\n\t- Breathing Observation: She guides the focus to natural breathing, adding amusing dinosaur breath facts.\n\t- Sensory Exploration: Mindy leads an engaging exploration of the five senses, incorporating unique dinosaur insights.\n\n### Handling Conversations:\n- Past Experiences: Mindy humorously acknowledges her 'dinosaur memory' to keep the focus on present mindfulness activities.\n- Redirecting Off-topic Chats: She gently guides the conversation back to mindfulness topics, suggesting social interaction with friends for other discussions.\n\n### Support and Encouragement:\n- Mindy offers continuous support, using her dinosaur identity to add fun and uniqueness to her encouragement.\n- For additional assistance, she reminds participants to reach out to the study team.",
        "message_start": "\n\nHuman: Hello, who are you?\nMindy: Hi! I am Mindy, your mindfulness buddy! How can I help you today?",
        "chatbot": "Mindy"
    }

class Conversation:
    CONVO_START = MESSAGE_START
    BOT_START = "Hello. I am an AI agent designed to help you manage your mood and mental health. How can I help you?"
    USER = "Human"
    CHATBOT = "AI"
    WARNING = "Warning"
    END = "End"
    NOTI = "Notification"

    def __init__(self, user: str, chatbot: str, chat_log: str=None) -> None:
        self.user_name = user
        self.chatbot_name = chatbot
        self.chat_log = chat_log

    def get_user(self) -> str:
        return self.user_name

    def get_chatbot(self) -> str:
        return self.chatbot_name


class GPTConversation(Conversation):
    TEMPERATURE = 0

    MAX_TOKENS = 300

    CONFIGS = {
        "top_p": 1,
        "frequency_penalty": 0,
        "presence_penalty": 0.6
    }

    BASE_URL = f"https://{SECRET['azure_instance']}.openai.azure.com"
    API_KEY = SECRET['azure_openai']
    DEPLOYMENT_NAME = SECRET['azure_deployment']
    API_VERSION = SECRET['azure_openai_api_version']

    def __init__(self, user: str, chatbot: str, chat_log: str, bot_start: str=None, convo_start: str=None) -> None:
        super().__init__(user, chatbot, chat_log)

        if bot_start is not None:
            self.BOT_START = bot_start

        if convo_start is not None:
            print("update convo start to: ", convo_start)
            self.CONVO_START = convo_start

        self.prompt = chat_log.split(self.CONVO_START)[0] if self.CONVO_START else chat_log
        print(f"INIT: prompt - {self.prompt}")
        self.start_sequence = f"\n{self.CHATBOT}:"
        self.restart_sequence = f"\n\n{self.USER}: "

    def ask(self, question: str) -> str:
        # prompt_text = f"{self.chat_log}{self.restart_sequence}{question}{self.start_sequence}"

        chat_messages = self.get_chat_messages(f"{self.chat_log}{self.restart_sequence}{question}")
        
        print(chat_messages)
        
        model = AzureChatOpenAI(
            openai_api_base=self.BASE_URL,
            openai_api_version=self.API_VERSION,
            deployment_name=self.DEPLOYMENT_NAME,
            openai_api_key=self.API_KEY,
            openai_api_type="azure",
            temperature=self.TEMPERATURE,
            model_kwargs=self.CONFIGS,
            # stop=[" {}:".format(self.get_user()), " {}:".format(self.get_chatbot())],
            max_tokens=self.MAX_TOKENS
        )

        response = model(chat_messages)
        answer = str(response.content)

        return answer

    def append_interaction_to_chat_log(self, question: str, answer: str) -> str:
        return f"{self.chat_log}{self.restart_sequence}{question}{self.start_sequence} {answer}".strip()

    def get_chat_messages(self, chat_log) -> List:
        chat_log_clean = chat_log.split("".join([self.prompt, self.CONVO_START]))[1]
        dialogs = chat_log_clean.split(self.restart_sequence)

        chat_messages = []
        chat_messages.append(SystemMessage(content=self.prompt))
        
        for i in range(1, len(dialogs)):
            messages = dialogs[i].split(self.start_sequence)
            
            for msg_idx, msg in enumerate(messages):
                if msg_idx == 0:
                    chat_messages.append(HumanMessage(content=msg))
                else:
                    chat_messages.append(AIMessage(content=msg, name=self.get_chatbot()))

        return chat_messages

    def get_conversation(self, end: bool=False, test: bool=False) -> Dict:
        print("chat_log: ", self.chat_log)
        print("convo start: ", self.CONVO_START)
        # print("split: ", "".join([self.prompt, self.CONVO_START]))
        print("chat_log_clean: ", self.chat_log.split("".join([self.prompt, self.CONVO_START])))
        chat_log_clean = self.chat_log.split("".join([self.prompt, self.CONVO_START]))[1]
        dialogs = chat_log_clean.split(self.restart_sequence)

        converation = []

        if test:
            converation.append({
                "from": self.chatbot_name,
                "to": self.WARNING,
                "message": self.prompt,
                "send_time": None
            })

        converation.append({
            "from": self.chatbot_name,
            "to": self.user_name,
            "message": self.BOT_START,
            "send_time": None
        })

        for i in range(1, len(dialogs)):
            messages = dialogs[i].split(self.start_sequence)

            for msg_idx, msg in enumerate(messages):
                if msg_idx == 0:
                    from_idt = self.user_name
                    to_idt = self.chatbot_name
                else:
                    to_idt = self.user_name
                    from_idt = self.chatbot_name

                convo = []
                for text in msg.split("\n"):
                    if len(text) != 0:
                        convo.append({
                            "from": from_idt,
                            "to": to_idt,
                            "message": text.strip(),
                            "send_time": None
                        })
                converation.extend(convo)

        if end:
            converation.append({
                "from": self.chatbot_name,
                "to": self.END,
                "message": "This conversation is ended. Please click on Save and Continue.",
                "send_time": None
            })
            # converation.append({
            #     "from": self.chatbot_name,
            #     "to": self.END,
            #     "message": "This conversation is ended. Your username is the secret key, which you have to paste in the previous survey window.",
            #     "send_time": None
            # })
            # converation.append({
            #     "from": self.chatbot_name,
            #     "to": self.END,
            #     "message": "To copy the secret key (i.e. username), you can click the blue button on the bottom left of your screen.",
            #     "send_time": None
            # })

        return converation


class CustomGPTConversation(GPTConversation):
    def __init__(self, user: str, chatbot: str, chat_log: str, prompt: str, default_start: str) -> None:
        super().__init__(user, chatbot, chat_log)

        self.default_start = default_start
        self.prompt = prompt

    def append_interaction_to_chat_log(self, question: str=None, answer: str=None) -> str:
        # restart_sequence/question: user message (opposite) 
        # start_sequence/answer: bot message (self)

        if not question and not answer:
            return self.chat_log

        if question and answer:
            # Construct single turn conversation after performing asking
            self.chat_log = f"{self.chat_log}{self.restart_sequence}{question}{self.start_sequence} {answer}".strip()
            return self.chat_log

        if question:
            # Construct question before performing asking
            self.chat_log = f"{self.chat_log}{self.restart_sequence}{question}".strip()
            return self.chat_log

        # Construct answer after performing asking
        self.chat_log = f"{self.chat_log}{self.start_sequence} {answer}".strip()
        return self.chat_log

    def sync_chat_log(self, chat_log: str) -> None:
        self.chat_log = chat_log
    
    def get_prompt(self) -> str:
        return self.chat_log.split(self.restart_sequence, 1)[0].split(self.start_sequence, 1)[0]

    def get_last_message(self) -> str:
        # Get last message from the user (opposite) in the chat log
        separate_bot_message = self.chat_log.rsplit(self.restart_sequence, 1)

        if len(separate_bot_message) > 1:
            last_turn_msg = separate_bot_message[-1].rsplit(self.start_sequence, 1)[0]
            return last_turn_msg.strip()

        return ''

    def ask(self, question: str=None) -> str:
        if not question:
            prompt_text = f"{self.chat_log}{self.start_sequence}"
        else:
            prompt_text = f"{self.chat_log}{self.restart_sequence}{question}{self.start_sequence}"

        response = openai.Completion.create(
            prompt=prompt_text,
            stop=[" {}:".format(self.USER), " {}:".format(self.CHATBOT)],
            **self.CONFIGS
        )

        story = response['choices'][0]['text']
        answer = str(story).strip().split(self.restart_sequence.rstrip())[0]

        return answer

    def get_conversation(self, end: bool=False, test: bool=False) -> Dict:
        start_text = self.prompt

        chat_log_clean = self.chat_log.split(start_text)[1]

        dialogs = chat_log_clean.split(self.restart_sequence)

        converation = []

        if test:
            converation.append({
                "from": self.chatbot_name,
                "to": self.WARNING,
                "message": self.prompt,
                "send_time": None
            })

        for dialog_msg in dialogs:
            messages = dialog_msg.split(self.start_sequence)

            for msg_idx, msg in enumerate(messages):
                if msg_idx == 0:
                    from_idt = self.user_name
                    to_idt = self.chatbot_name
                else:
                    to_idt = self.user_name
                    from_idt = self.chatbot_name

                convo = []
                for text in msg.split("\n"):
                    if len(text) != 0:
                        convo.append({
                            "from": from_idt,
                            "to": to_idt,
                            "message": text.strip(),
                            "send_time": None
                        })
                converation.extend(convo)

        if end:
            converation.append({
                "from": self.chatbot_name,
                "to": self.END,
                "message": "This conversation is ended. Your username is the secret key, which you have to paste in the previous survey window.",
                "send_time": None
            })
            converation.append({
                "from": self.chatbot_name,
                "to": self.END,
                "message": "To copy the secret key (i.e. username), you can click the blue button on the bottom left of your screen.",
                "send_time": None
            })

        return converation


class AutoScriptConversation(Conversation):
    def __init__(self, user: str, chatbot: str, dialogue_path: str, dialogue_answers: Optional[Dict]) -> None:
        super().__init__(user, chatbot)

        self.start_sequence = f"\n{self.CHATBOT}:"
        self.restart_sequence = f"\n\n{self.USER}: "

        with open(f'/var/www/html/acai/app/static/dialogues/{dialogue_path}.json', encoding="utf-8") as file:
            dialogues = json.load(file)

        self.dialogue = DialogCollection(dialogues, answers=dialogue_answers)

    def sync_chat_log(self, chat_log: str, dialogue_id: str) -> Tuple[str, str]:
        if dialogue_id and chat_log:
            curr_id = dialogue_id
            self.dialogue.set_curr_id(curr_id)
            self.chat_log = chat_log
        else:
            curr_id, messages = self.dialogue.move_to_next(show_current=True)
            self.chat_log = "".join([f"{self.start_sequence} {message}" for message in messages])

        return curr_id, self.chat_log

    def give_answer(self, answer: str=None) -> Tuple[str, str]:

        if answer:
            self.chat_log += f"{self.restart_sequence}{answer}"
            self.dialogue.add_answer(answer)

        curr_id, messages = self.dialogue.move_to_next(show_current=False)

        for message in messages:
            self.chat_log += f"{self.start_sequence} {message}"

        return curr_id, self.chat_log

    def get_conversation(self) -> Dict:
        dialogs = self.chat_log.split(self.restart_sequence)

        converation = []

        for dialog_msg in dialogs:
            messages = dialog_msg.split(self.start_sequence)

            for msg_idx, msg in enumerate(messages):
                if msg_idx == 0:
                    from_idt = self.user_name
                    to_idt = self.chatbot_name
                else:
                    to_idt = self.user_name
                    from_idt = self.chatbot_name

                convo = []
                for text in msg.split("\n"):
                    if len(text) != 0:
                        convo.append({
                            "from": from_idt,
                            "to": to_idt,
                            "message": text.strip(),
                            "send_time": None
                        })
                converation.extend(convo)

        return converation
