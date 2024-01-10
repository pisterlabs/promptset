from langchain.chains import ConversationChain
from langchain.llms.openai import OpenAI
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory


def converse(person_1_description, person_2_description):
    llm_one = OpenAI(
        temperature=0.9,
        # openai_api_key="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",
    )
    llm_two = OpenAI(
        temperature=0.9,
        # openai_api_key="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",
    )
    # system_description = """ You are having a conversation with another person whom you are meeting for the first time.
    #     You are not an AI, but rather an individual with the following description. Match the person in their tone of
    #     voice. Be friendly, but not overly friendly.
    #     {}
    #     """
    system_description = """ You are not an AI, but an individual with the following description. When
        you converse, you must pretend as if you were this individual and mirror their personality, their hobbies,
        their goals, etc. 
        DESCRIPTION:
        {}
        You are meeting another person for the first time. Have a casual conversation with them. Be friendly and
        inquisitive, but not overly inquisitive. Do not regurgitate facts about yourself. Do not repeat yourself.
        When you are ready to end the conversation, say "Goodbye"
        """
    template_rest = """
        Current conversation:
        {history}
        Human: {input}
        AI:"""
    return _converse(
        llm_one,
        llm_two,
        person_1_description,
        person_2_description,
        system_description=system_description,
        template_rest=template_rest,
    )


def _converse(
    llm_one,
    llm_two,
    person_1_description,
    person_2_description,
    system_description,
    template_rest,
):
    """
    Given two LLMs, this returns the conversation.

    :param llm_one:
    :param llm_two:
    :return:
    """
    conversation_1_prompt = PromptTemplate(
        input_variables=["history", "input"],
        template=system_description.format(person_1_description) + template_rest,
    )
    conversation_2_prompt = PromptTemplate(
        input_variables=["history", "input"],
        template=system_description.format(person_2_description) + template_rest,
    )

    conversation_one = ConversationChain(
        llm=llm_one,
        memory=ConversationBufferMemory(),
        prompt=conversation_1_prompt,
    )
    conversation_two = ConversationChain(
        llm=llm_two,
        memory=ConversationBufferMemory(),
        prompt=conversation_2_prompt,
    )
    output_two = "Hey there!"
    print("2: ", output_two)
    for i in range(3):
        output_one = conversation_one.predict(
            input=output_two,
        )
        print("1: ", output_one)
        if "Goodbye" in output_one:
            break
        output_two = conversation_two.predict(
            input=output_one,
        )
        print("2: ", output_two)
        if "Goodbye" in output_two:
            break
    return conversation_one


if __name__ == "__main__":
    # person_1_description = "--------- Passionate about computer vision and animal rights----- BS/MS student at Stanford----- Enjoys reading, hiking, and playing video games----- Values honesty, integrity, and hard work----- Has a strong sense of justice and fairness----- Likes to explore new ideas and technologies----- Enjoys learning new things and meeting new people"
    person_1_description = "'\n1. Personality: Outgoing, adventurous, creative\n2. Values: Family, sports, books, music\n3. Background: 16-year-old from a small town in Indiana, from a big family, plays soccer, loves books, foodie, plays guitar\n4. Age: 16\n5. Name: Sarah\n6. Hobbies: Soccer, reading, cooking, playing guitar\n7. Goals: Balance sports, books, and baking\n8. Political Leanings: Unknown'"
    # person_2_description = (
    #     "'----1. Personality: Committed to making a positive impact on the world, believes in the power of change through community involvement and political participation.----2. Values: Hard work, education, and community involvement.----3. Background: Born in Honolulu, Hawaii on August 4, 1961, with a Kenyan father and a Kansan mother. 44th President of the United States from 2009 to 2017.----5. Name: Barack Obama----6. Hobbies: Unknown----7. Goals: To bridge divisions and work toward a more inclusive and equitable America.----8. Political Leanings: Democratic'",
    # )
    person_2_description = "'\n1. Personality: Grease in his veins, loves the outdoors, avid fisherman, loves the Alaskan lifestyle\n2. Values: Family, hard work, adventure\n3. Background: 42-year-old from Wasilla, Alaska, works as a mechanic, two children\n4. Age: 42\n5. Name: Mike\n6. Hobbies: Fishing, snowmobiling, mechanic work\n7. Goals: To provide for his family and enjoy the Alaskan wilderness\n8. Political Leanings: Unknown'"

    converse(person_1_description, person_2_description)
