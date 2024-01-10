from openai import OpenAI
import streamlit as st
from format import print_double_space, submit_button, show_result
from streamlit import session_state

functions_file_path = "./rhetorical_functions_few.txt"
with open(functions_file_path, "r") as functions_file:
    rhetorical_functions_few = functions_file.read()

CoT_Example_path = "./CoT_Example.txt"
with open(CoT_Example_path, "r") as CoT_Example_file:
    CoT_Example = CoT_Example_file.read()


def process_article(article_content):
    # 預設立場職業 - 語言學家
    CONTENT_linguist = "I want you to act as if you are a linguist who are familiar with functional language which is meta discourse and not the content itself. Phrases in functional language typically appear at the beginning of a sentence, are high in frequency with common and general words."
    # 說明規則
    CONTENT_functions = f"""Here are 12 types of rhetorical functions and their examples.
{rhetorical_functions_few}

Do the following steps:
"""
    # 辨識規則
    CONTENT_identify = """Read the following article and identify the word or phrases realizing rhetorical functions for each sentence.
Notice:
    1. Phrases should use less than 5 words.
    2. A sentence may have no phrase using rhetorical function.
    3. A sentence may have two or more phrases using rhetorical functions.
"""
    # 表格格式
    CONTENT_table = """Make a table, which containing the sentence number, sentence, phrase, and rhetorical function.
Notice:
    1. The table should have 4 columns: Sentence Number, Sentence, Phrase, Rhetorical Function.
    2. The rhetorical Function content should be written as "<Alphabet>. <Big Title of Function>".
    3. Put all sentences in the table. If a sentence has no phrase realizing rhetorical functions, leave the phrase and rhetorical function columns blank.
"""
    # 輸出表格
    CONTENT_output = """Must output only the table. Do not output any other message."""
    # 一步步做
    CONTENT_step_by_step = """Let's think and do it step by step."""
    # 思考鏈範例
    CONTENT_CoT_Example = f"""Given an article:
{CoT_Example}

Output:
Find the phrases sentence by sentence.

Sentence 1: I certainly agree that travelling in a group with a tour guide is the best option.\n
    The phrase, [[I certainly agree that]], expresses the author's ideas about traveling directly and clearly. It is the rhetorical function E. Expressing Personal Opinions.\n
    The phrase, [[certainly]], shows a high probability of agreeing that traveling is the best option. It is the rhetorical function F. Expressing Possibility/Certainty POSS.\n
Sentence 2: Travelling outside of the place where you live is one of the most exiting and enjoyable things to do in most peoples lives.\n
    There is no phrase realizing rhetorical function in this sentence.\n
Sentence 3: People tend to take vacations to get away from the rigours of daily life.\n
    The phrase, [[tend to]], shows a certain degree of probability about people's preferences. It is the rhetorical function F. Expressing Possibility/Certainty POSS.\n
Sentence 4: Most people would certainly like to go alone or with an individual family if the choice existed.\n
    The phrase, [[certainly like to]], shows a certain degree of probability about people's preferences. It is the rhetorical function F. Expressing Possibility/Certainty POSS.\n
Sentence 5: However most places that people consider are far away and exotic in nature, not to ignore the fact that they could be dangerous too.\n
    The phrase, [[However]], expresses a cautious concession between being alone and safety. It is the rhetorical function G. Introducing a Concession.\n
Sentence 6: Take for example the Bali in Indonesia.\n
    The phrase, [[Take for example]], gives an example of previous viewpoints. It is the rhetorical function C. Introducing an Example.\n
Sentence 7: The recent terrorist strike in Bali killed a lot of people, mostly vacationeers.\n
    There is no phrase realizing rhetorical function in this sentence.\n
Sentence 8: Going in a group offers a lot of benifits to a traveller.\n
    There is no phrase realizing rhetorical function in this sentence.\n
Sentence 9: Most airlines and hotels offer group booking discounts as well as preferential service.\n
    There is no phrase realizing rhetorical function in this sentence.\n
Sentence 10: Travelling in a group also makes you feel secure rather than travelling alone.\n
    The phrase, [[also]], adds more information about the benefits of group travel. It is the rhetorical function A. Adding Information.\n
Sentence 11: Travelling in groups would be specially advantageous to people with kids as I believe kids are better behaved and more entertained when they are in a group.\n
    The phrase, [[I believe]], expresses the author's ideas about kids group travel directly and clearly. It is the rhetorical function E. Expressing Personal Opinions.\n
Sentence 12: Going in a group with a tour guide would also help people with disabilities.\n
    The phrase, [[also]], adds more information about the benefits of group travel. It is the rhetorical function A. Adding Information.\n
Sentence 13: Also, why do something when you can have someone do it for you?\n
    The phrase, [[Also,]], adds more information about the benefits of group travel. It is the rhetorical function A. Adding Information.\n
Sentence 14: This is the type of question most people would ask when going out on a trip.\n
    There is no phrase realizing rhetorical function in this sentence.\n
Sentence 15: Instead of spending time and researching about a particular destination, it would be better to employ a local expert ( tour guide ) to guide you to the right places and explain the history and culture of that particular place.\n
    The phrase, [[Instead of]], expresses a cautious concession between researching by self and being guided by others. It is the rhetorical function G. Introducing a Concession.\n
    The phrase, [[it would be better]], indirectly expresses the author's thought that group travel is better than solo travel. It is the rhetorical function E. Expressing Personal Opinions.\n
Sentence 16: A tour guide would also be helpful incase there is an emergency and you might need to contact the local authorities or need any kind of help.\n
    The phrase, [[also]], adds more information about the benefits of group travel. It is the rhetorical function A. Adding Information.\n
Sentence 17: Also with their experience most tour guides are able to guide us to the right spots and destinations, making the most of our vacation.\n
    The phrase, [[Also]], adds more information about the benefits of group travel. It is the rhetorical function A. Adding Information.\n

Convert all sentences into a table with 4 columns: Sentence Number, Sentence, Phrase, Rhetorical Function.

| Sentence Number | Sentence | Phrase | Rhetorical Function |
| -------------- | -------- | ------ | ------------------- |
| 1 | I certainly agree that travelling in a group with a tour guide is the best option. | I certainly agree that | E. Expressing Personal Opinions |
| 1 | I certainly agree that travelling in a group with a tour guide is the best option. | certainly | F. Expressing Possibility/Certainty POSS |
| 2 | Travelling outside of the place where you live is one of the most exiting and enjoyable things to do in most peoples lives. |  |  |
| 3 | People tend to take vacations to get away from the rigours of daily life. | tend to | F. Expressing Possibility/Certainty POSS |
| 4 | Most people would certainly like to go alone or with an individual family if the choice existed. | certainly like to | F. Expressing Possibility/Certainty POSS |
| 5 | However most places that people consider are far away and exotic in nature, not to ignore the fact that they could be dangerous too. | However | G. Introducing a Concession |
| 6 | Take for example the Bali in Indonesia. | Take for example | C. Introducing an Examples |
| 7 | The recent terrorist strike in Bali killed a lot of people, mostly vacationeers. |  |  |
| 8 | Going in a group offers a lot of benifits to a traveller. |  |  |
| 9 | Most airlines and hotels offer group booking discounts as well as preferential service. |  |  |
| 10 | Travelling in a group also makes you feel secure rather than travelling alone. | also | A. Adding Information |
| 11 | Travelling in groups would be specially advantageous to people with kids as I believe kids are better behaved and more entertained when they are in a group. | I believe | E. Expressing Personal Opinions |
| 12 | Going in a group with a tour guide would also help people with disabilities. | also | A. Adding Information |
| 13 | Also, why do something when you can have someone do it for you? | Also, | A. Adding Information |
| 14 | This is the type of question most people would ask when going out on a trip. |  |  |
| 15 | Instead of spending time and researching about a particular destination, it would be better to employ a local expert ( tour guide ) to guide you to the right places and explain the history and culture of that particular place. | Instead of | G. Introducing a Concession |
| 15 | Instead of spending time and researching about a particular destination, it would be better to employ a local expert ( tour guide ) to guide you to the right places and explain the history and culture of that particular place. | it would be better | E. Expressing Personal Opinions |
| 16 | A tour guide would also be helpful incase there is an emergency and you might need to contact the local authorities or need any kind of help. | also | A. Adding Information |
| 17 | Also with their experience most tour guides are able to guide us to the right spots and destinations, making the most of our vacation. | Also | A. Adding Information |
"""
    # RESPONSE_1 = "OK. I will do it step by step and strictly follow the specified format above. Please give me more articles."
    CONTENT_test_article = f"""Given another article:
{article_content}

Output:
"""

    model = "gpt-4"
    # model="gpt-3.5-turbo"

    messages_1 = [
        {"role": "user", "content": CONTENT_linguist},
        {"role": "user", "content": CONTENT_functions},
        {"role": "user", "content": CONTENT_identify},
        {"role": "user", "content": CONTENT_table},
        # {'role': 'user', 'content': CONTENT_output},
        {"role": "user", "content": CONTENT_step_by_step},
        {"role": "user", "content": CONTENT_CoT_Example},
        {"role": "user", "content": CONTENT_test_article},
    ]

    st.subheader("**Classification of Rhetorical**")

    write_rhetorical_functions()

    # massenge
    submit_msg = submit_button()
    submit_prompt = st.button(submit_msg[session_state.submit_revise_query])
    result = show_result()
    st.subheader(result[session_state.submit_revise_query])
    print_lines = """"""
    # result
    if submit_prompt:
        # Get ChatGPT answer
        client = OpenAI()
        response_1 = client.chat.completions.create(
            model=model,
            messages=messages_1,
            temperature=0.7,
            # max_tokens=1000
        )
        response_table = response_1.choices[0].message.content

        fixed_sentence = getfixedsentence(article_content, response_table)

        # Using errant to compare and produce edited articale
        # fixed_sentence = match_tokens_from_errant(article_content, fixed_sentence)
        # st.write(article_content)
        # st.write(fixed_sentence)

        # Display double space format
        print_lines = print_double_space(fixed_sentence)

        tabel_title = {
            "EN": "**Analysis for each sentence and Explanation Table:**",
            "TN": "**以下為分析與表格：**",
            "JP": "**分析と表：**",
        }
        st.subheader(tabel_title[session_state.submit_revise_query])
        st.write(response_table)

        # Original fixed sentence
        # with st.expander('Original Fixed Sentence'):
        # st.write(fixed_sentence)
        # All content of reponse
        # with st.expander('Whole Response'):
        # st.write(response_1)
    return print_lines


def write_rhetorical_functions():
    functions_file_path = "./rhetorical_functions.txt"

    with st.expander("Rhetorical Function", expanded=True):
        functions_file = open(functions_file_path, "r")
        for sentence in functions_file:
            st.write(sentence)


def getfixedsentence(article_content, response_table):
    # 預設立場職業 - Assistant
    CONTENT_Assistant = """You are a helpful, pattern-following assistant."""
    # 轉換表格
    CONTENT_Convert = """Do the following:
Converts a article with a table which containing the sentence number, sentence, phrase, and rhetorical function into a article with phrase in square brackets and rhetorical functions in curly brackets:
Notice:
    1. The table have 4 columns: Sentence Number, Sentence, Phrase, Rhetorical Function.
    2. Mark each phrase with square brackets (i.e., "[-", "-]"), and insert the corresponding rhetorical function with curly brackets  (i.e., "{{+", "+}}") right before each phrase.
"""
    # 思考鏈範例
    CONTENT_Convert_CoT_Example = """Given an article and its table:
{CoT_Example}

| Sentence Number | Sentence | Phrase | Rhetorical Function |
| -------------- | -------- | ------ | ------------------- |
| 1 | I certainly agree that travelling in a group with a tour guide is the best option. | I certainly agree that | E. Expressing Personal Opinions |
| 1 | I certainly agree that travelling in a group with a tour guide is the best option. | certainly | F. Expressing Possibility/Certainty POSS |
| 2 | Travelling outside of the place where you live is one of the most exiting and enjoyable things to do in most peoples lives. |  |  |
| 3 | People tend to take vacations to get away from the rigours of daily life. | tend to | F. Expressing Possibility/Certainty POSS|
| 4 | Most people would certainly like to go alone or with an individual family if the choice existed. | certainly like to | F. Expressing Possibility/Certainty POSS|
| 5 | However most places that people consider are far away and exotic in nature, not to ignore the fact that they could be dangerous too. | However | G. Introducing a Concession
| 6 | Take for example the Bali in Indonesia. | Take for example | C. Introducing an Examples
| 7 | The recent terrorist strike in Bali killed a lot of people, mostly vacationeers. |  |  |
| 8 | Going in a group offers a lot of benifits to a traveller. |  |  |
| 9 | Most airlines and hotels offer group booking discounts as well as preferential service. |  |  |
| 10 | Travelling in a group also makes you feel secure rather than travelling alone. | also | A. Adding Information |
| 11 | Travelling in groups would be specially advantageous to people with kids as I believe kids are better behaved and more entertained when they are in a group. | I believe | E. Expressing Personal Opinions |
| 12 | Going in a group with a tour guide would also help people with disabilities. | also | A. Adding Information |
| 13 | Also, why do something when you can have someone do it for you? | Also, | A. Adding Information |
| 14 | This is the type of question most people would ask when going out on a trip. |  |  |
| 15 | Instead of spending time and researching about a particular destination, it would be better to employ a local expert ( tour guide ) to guide you to the right places and explain the history and culture of that particular place. | Instead of | G. Introducing a Concession |
| 15 | Instead of spending time and researching about a particular destination, it would be better to employ a local expert ( tour guide ) to guide you to the right places and explain the history and culture of that particular place. | it would be better | E. Expressing Personal Opinions |
| 16 | A tour guide would also be helpful incase there is an emergency and you might need to contact the local authorities or need any kind of help. | also | A. Adding Information |
| 17 | Also with their experience most tour guides are able to guide us to the right spots and destinations, making the most of our vacation. | Also | A. Adding Information |

Output:
{+E. Expressing Personal Opinions+} [-I certainly agree that-] travelling in a group with a tour guide is the best option. Travelling outside of the place where you live is one of the most exiting and enjoyable things to do in most peoples lives. People {+F. Expressing Possibility/Certainty POSS+} [-tend to-] take vacations to get away from the rigours of daily life. Most people would {+F. Expressing Possibility/Certainty POSS+} [-certainly like to-] go alone or with an individual family if the choice existed. {+G. Introducing a Concession+} [-However-] most places that people consider are far away and exotic in nature, not to ignore the fact that they could be dangerous too. {+C. Introducing an Examples+} [-Take for example-] the Bali in Indonesia. The recent terrorist strike in Bali killed a lot of people, mostly vacationeers.

Going in a group offers a lot of benifits to a traveller. Most airlines and hotels offer group booking discounts as well as preferential service. Travelling in a group {+A. Adding Information+} [-also-] makes you feel secure rather than travelling alone. Travelling in groups would be specially advantageous to people with kids as {+E. Expressing Personal Opinions+} [-I believe-] kids are better behaved and more entertained when they are in a group. Going in a group with a tour guide would {+Adding Information+} [-also-] help people with disabilities. {+A. Adding Information+} [-Also,-] why do something when you can have someone do it for you? This is the type of question most people would ask when going out on a trip. {+G. Introducing a Concession+} [-Instead of-] spending time and researching about a particular destination, {+E. Expressing Personal Opinions+} [-it would be better-] to employ a local expert ( tour guide ) to guide you to the right places and explain the history and culture of that particular place. A tour guide would {+Adding Information+} [-also-] be helpful incase there is an emergency and you might need to contact the local authorities or need any kind of help. {+A. Adding Information+} [-Also-] with their experience most tour guides are able to guide us to the right spots and destinations, making the most of our vacation.
"""

    CONTENT_Convert_test_article = f"""Given another article and its table:
{article_content}

{response_table}

Output:
"""

    model = "gpt-4"
    # model="gpt-3.5-turbo"

    messages_2 = [
        {"role": "user", "content": CONTENT_Assistant},
        {"role": "user", "content": CONTENT_Convert},
        {"role": "user", "content": CONTENT_Convert_CoT_Example},
        {"role": "user", "content": CONTENT_Convert_test_article},
    ]

    client = OpenAI()
    response_2 = client.chat.completions.create(
        model=model,
        messages=messages_2,
        temperature=0.7,
        # max_tokens=1000
    )

    fixed_sentence = response_2.choices[0].message.content

    return fixed_sentence
