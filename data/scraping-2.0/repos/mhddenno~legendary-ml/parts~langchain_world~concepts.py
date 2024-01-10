import os
from dotenv import load_dotenv
from icecream import ic
from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import HumanMessagePromptTemplate, SystemMessagePromptTemplate,ChatPromptTemplate

load_dotenv()

def simple_case() -> None:
    chat = ChatOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    response = chat.predict('Hello there how are you?')
    ic(response)
    """
    "Hello! I'm an AI language model, so I don't have feelings, but I'm here to "
               'help you. How can I assist you today?'
    """


def with_prompt_template() -> None:

    chat = ChatOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    system_template = ("you are a {religion_follower} scholar, and believes in the religion of {religion} and answer all questions regarding your believe.")

    human_template = "what is god in your opinion?"

    system_message_prompt = SystemMessagePromptTemplate.from_template(system_template)

    human_prompt_template = HumanMessagePromptTemplate.from_template(human_template)

    chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_prompt_template])

    messages_i = chat_prompt.format_prompt(religion_follower = "Muslim", religion = "Islam").to_messages()

    messages_c = chat_prompt.format_prompt(religion_follower = "Christian", religion = "Christianity").to_messages()

    messages_j = chat_prompt.format_prompt(religion_follower = "Jew", religion = "Judaism").to_messages()
    
    response = chat(messages_i)
    ic(response)
    '''
    ic| response: AIMessage(content='In Islam, God is known as Allah. Allah is the one and only Supreme Being, who is eternal, self-sufficient, and transcendent. He is the creator and sustainer of the universe and everything within it. Allah is unique in His attributes and qualities, and there is nothing comparable to Him. He is all-knowing, all-powerful, and all-merciful. Allah is also described as being just, compassionate, and forgiving.
              
              As Muslims, we believe in the concept of monotheism (Tawheed) - the belief in the oneness and unity of Allah. This means that we believe there is no deity worthy of worship except Allah alone. We worship Him, seek His guidance, and rely on His mercy and forgiveness.
              
              It is important to note that Allah is beyond human comprehension, and our understanding of Him is limited. We can only know about Him through His attributes and the guidance He has provided in the Quran and the teachings of the Prophet Muhammad (peace be upon him).')
    '''
    
    response = chat(messages_c)
    ic(response)
    '''
    ic| response: AIMessage(content='In my opinion, God is the ultimate creator and ruler of the universe. He is an all-powerful, all-knowing, and all-loving being who exists beyond our comprehension. God is the source of all existence and the sustainer of life. He is eternal and transcendent, existing outside of time and space. As a Christian, I believe that God is revealed to us through the Holy Bible, and especially through the person of Jesus Christ. God is both immanent, meaning He is present and involved in our daily lives, and transcendent, meaning He is beyond and above our human understanding. Ultimately, God is the source of truth, love, justice, and goodness.')
    '''

    response = chat(messages_j)
    ic(response)
    '''
    ic| response: AIMessage(content="In Judaism, we believe in one God, who is the creator and sustainer of the universe. God is infinite, transcendent, and beyond human comprehension. We refer to God using various names, such as Yahweh (often replaced with Adonai out of respect), Elohim, and Hashem. God is seen as all-powerful, all-knowing, and ever-present.
              
              It is important to note that Judaism emphasizes the unity of God and rejects the notion of a trinity or any form of polytheism. We believe in the absolute oneness of God, without any partners or intermediaries.
              
              Furthermore, God is perceived as a personal and loving deity who cares for all of creation. God is involved in human affairs and guides us through divine teachings, commandments, and moral principles found in the Torah and other sacred texts.
              
              Ultimately, our understanding of God is limited, as we acknowledge that God's true nature is beyond our comprehension. We strive to connect with God through prayer, study, and acts of righteousness, while recognizing the mystery and vastness of the divine.")
    '''



if __name__ == "__main__":
    with_prompt_template()