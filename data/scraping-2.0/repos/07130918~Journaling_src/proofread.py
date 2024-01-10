from datetime import datetime

from dotenv import load_dotenv
from langchain import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import (ChatPromptTemplate,
                                    HumanMessagePromptTemplate,
                                    SystemMessagePromptTemplate)

JOURNALING_DIR = "/Users/satokota/Desktop/English/Journaling"
NOW = datetime.now()
CURRENT_YEAR = NOW.strftime("%Y")
CURRENT_MONTH_NAME = NOW.strftime("%B")
TODAY = (f"{CURRENT_YEAR}/{CURRENT_MONTH_NAME}/{str(NOW.strftime('%d')).zfill(2)}")
# 学習したい言語
LANGUAGE = "English"


def main():
    """proofread journaling

    Raises:
        FileNotFoundError: if journaling file is not found

    Returns:
        str: response from GPT-3
        None: if journaling file is not found
    """
    print("Loading env...")
    load_dotenv()
    print("Proofreading...")

    try:
        with open(f"{JOURNALING_DIR}/{TODAY}.txt", "r") as f:
            journal = f.read()
    except FileNotFoundError as e:
        print(e)
        return None

    system_template = "You are a helpful and capable assistant who can behave as a native {language} speaker."
    user_template = """
    I'm looking for someone who can play the role of a native {language} speaker and teacher.
    I'm Japanese and currently learning {language} as my second language.
    I've written a journal and I'd appreciate it if you could review it and correct it to sound more natural and accurate.

    Also, You must write each sentence on a new line.

    For example, if the sentences are like this:
    ###
    I want to improve my English skills. Besides, I would love to have friends who speak fluent English.
    ###

    I'd like your response to be something like this:
    ###
    I'm eager to enhance my English proficiency.

    Additionally, I'd also really enjoy making friends with those who are fluent in English.
    ###

    Today's journal: {journal}
    Your answer:
    """

    # temperatureは値を変更していろいろ試したけど、0.8が良さそう。
    gpt = ChatOpenAI(model="gpt-4", temperature=0.8, client=None)
    system_prompt = SystemMessagePromptTemplate.from_template(system_template)
    user_prompt = HumanMessagePromptTemplate.from_template(user_template)
    chat_prompt = ChatPromptTemplate.from_messages([system_prompt, user_prompt])

    chain = LLMChain(llm=gpt, prompt=chat_prompt)
    response = chain.run(language=LANGUAGE, journal=journal)

    print("-" * 20)
    print(response)
    print("-" * 20)

    return response


if __name__ == '__main__':
    main()
