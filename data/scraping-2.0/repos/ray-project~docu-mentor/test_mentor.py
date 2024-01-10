from main import mentor, ANYSCALE_API_ENDPOINT
import openai
import os
import pytest
import re


def gpt4_evaluator(answers):
    gpt4_instructions = """
        You are a helpful assistant.
        Another system has been tasked with improving syntax, grammar, punctuation, style, etc.
        of the following <content>.
        The <content> is in JSON format and contains file name keys and text values.
        Be thorough, but not overly critical. If the content is good, there's no need to comment on it.
    """
    extra_instructions= """If the <content> is good as is, simply return "ok".
    Otherwise point out mistakes that have been missed.
    At the very end, give an assessment of what percentage of answers were sufficient.
    E.g., if only 2 out of 50 answers needed improvement, success rate of 96%.
    It's important to return this success rate as string."""
    res = mentor(content=answers, model="gpt-4", system_content=gpt4_instructions, extra_instructions=extra_instructions)
    return res["choices"][0]["message"]["content"]



def test_mentor_sentences(flawed_sentences):
    # Set keys to Anyscale Endpoints usage
    openai.api_base = ANYSCALE_API_ENDPOINT
    openai.api_key = os.environ.get("ANYSCALE_API_KEY")

    # mentor test data with doc-sanity models hosted by Anyscale Endpoints
    corrected_sentences = mentor(flawed_sentences)["choices"][0]["message"]["content"]
    print(corrected_sentences)

    # Redirect to OpenAI to use GPT-4 as evaluator
    OPENAI_API_ENDPOINT="https://api.openai.com/v1"
    openai.api_base = OPENAI_API_ENDPOINT
    openai.api_key = os.environ.get("GPT4_API_KEY")

    # Print the evaluation results
    gpt_sentence_eval = gpt4_evaluator(corrected_sentences)
    print(gpt_sentence_eval)

    pattern = r'\d+%'
    matches = re.findall(pattern, gpt_sentence_eval)
    if matches:
        percentage = float(matches[0][:-1])
        assert percentage > 80


def test_mentor_paragraphs(flawed_paragraphs):
    # Set keys to Anyscale Endpoints usage
    openai.api_base = ANYSCALE_API_ENDPOINT
    openai.api_key = os.environ.get("ANYSCALE_API_KEY")

    # mentor test data with doc-sanity models hosted by Anyscale Endpoints
    corrected_paragraphs = mentor(flawed_paragraphs)["choices"][0]["message"]["content"]
    print(corrected_paragraphs)

    # Redirect to OpenAI to use GPT-4 as evaluator
    OPENAI_API_ENDPOINT="https://api.openai.com/v1"
    openai.api_base = OPENAI_API_ENDPOINT
    openai.api_key = os.environ.get("GPT4_API_KEY")

    # Print the evaluation results
    gpt_paragrap_eval = gpt4_evaluator(corrected_paragraphs)
    print(gpt_paragrap_eval)

    # Doc-sanity should match the expectations of GPT-4 80% of the time,
    # as GPT tends to be fairly critical for longer paragraphs overall.
    pattern = r'\b\d+(\.\d+)?%\b'
    matches = re.findall(pattern, gpt_paragrap_eval)
    if matches:
        percentage = float(matches[0][:-1])
        assert percentage > 80


@pytest.fixture
def flawed_sentences():
    """Result of prompting GPT-4: I want to write a test in Python that takes a dictionary as input.
    The values should contain complete English sentences that have some deficiency,
    e.g. in punctuation, syntax, grammar, style or otherwise.
    The keys should mock file names. Give me such a dictionary with 50 diverse and
    interesting examples of bad writing."""
    return {
        "file_001.txt": "its a lovely day isn't.",
        "file_002.txt": "She likes apples, bananas and oranges strawberries.",
        "file_003.txt": "Him and me went to the store yesterday.",
        "file_004.txt": "Whose that guy standing over their?",
        "file_005.txt": "I has been to Paris last summer.",
        "file_006.txt": "They're cat is sitting on the roof.",
        "file_007.txt": "The boy, that won the race, was happy.",
        "file_008.txt": "If I would of known, I would of come.",
        "file_009.txt": "She said that she don't like ice cream.",
        "file_010.txt": "She's bag is red in colour.",
        "file_011.txt": "You need to try hard, however, not too hard.",
        "file_012.txt": "My sister, she is doctor.",
        "file_013.txt": "He didn't did his homework.",
        "file_014.txt": "He's book is on the shelf which it is green.",
        "file_015.txt": "You doing good today.",
        "file_016.txt": "Me and her likes to read.",
        "file_017.txt": "Please your shoes off.",
        "file_018.txt": "If I would've knew, I wouldn't have came.",
        "file_019.txt": "I seen him at the park yesterday.",
        "file_020.txt": "She can sings very well.",
        "file_021.txt": "They was playing football.",
        "file_022.txt": "He ate a lot of food yesterday he was full.",
        "file_023.txt": "Where are the keys at?",
        "file_024.txt": "Me want some ice cream.",
        "file_025.txt": "She needs to study she has an exam.",
        "file_026.txt": "It's raining but however I'll go outside.",
        "file_027.txt": "He plays good basketball.",
        "file_028.txt": "There are less apples in the basket than oranges.",
        "file_029.txt": "The more quicker you finish, the more better it is.",
        "file_030.txt": "I'll tell him when he will arrive.",
        "file_031.txt": "She should of studied harder.",
        "file_032.txt": "I'm very much tired.",
        "file_033.txt": "This is more better than that.",
        "file_034.txt": "You should drink less coffees.",
        "file_035.txt": "Each of the student have a book.",
        "file_036.txt": "Neither John or Peter was available.",
        "file_037.txt": "You should try to do it more faster.",
        "file_038.txt": "I'm loving this ice cream.",
        "file_039.txt": "She is elder than me.",
        "file_040.txt": "I haven't no money.",
        "file_041.txt": "The cat, it's on the roof.",
        "file_042.txt": "Either you're right or your wrong.",
        "file_043.txt": "He said he wants to go to they're house.",
        "file_044.txt": "I'm not sure whose car this is is it your's?",
        "file_045.txt": "It's a nice weather, isn't?",
        "file_046.txt": "She neither likes coffee nor tea.",
        "file_047.txt": "He can to drive.",
        "file_048.txt": "My friend he is a engineer.",
        "file_049.txt": "This movie is most funniest one I've seen.",
        "file_050.txt": "I ain't doing nothing right now.",
    }

@pytest.fixture
def flawed_paragraphs():
    """Result of prompting GPT-4: Give me a dict with 10 items, but the values should now contain
    full paragraphs of text with subtle stylistic errors"""
    return {
        "para_001.txt":
        """
        When I was a kid, I always wanted to visit Paris. You see, for as long as I can remember, Paris was on my bucket list. The thing is, Paris always felt like a dream, an unreachable goal.
        """,

        "para_002.txt":
        """
        Today's society is characterized by the fact that it's heavily dependent on technology. In fact, it can be argued that technology is now an integral part of our daily lives. And this is not necessarily a bad thing.
        """,

        "para_003.txt":
        """
        The importance of reading books is very underrated in our generation. By saying this, I mean that more people should engage in reading. Reading expands the mind, in other words.
        """,

        "para_004.txt":
        """
        It was a dark, stormy night. The winds howled, trees swayed, and in essence, it felt like nature was angry. You could say, it was one of those nights you'd rather stay indoors.
        """,

        "para_005.txt":
        """
        Jane has always been passionate about music. The thing about Jane is that she loves playing the guitar. Moreover, she's been playing since she was 10.
        """,

        "para_006.txt":
        """
        While walking in the forest, I suddenly realized its beauty. The forest was tranquil, calm, and in a way, it felt like time had stopped. In short, it was an experience I won't forget.
        """,

        "para_007.txt":
        """
        Climate change is a topic that's of paramount importance. This is to say, we should all pay more attention to this pressing issue. Especially when considering its implications.
        """,

        "para_008.txt":
        """
        The city is bustling with life. Everywhere you look, there's something happening. For instance, vendors selling, kids playing, and, to put it simply, life unfolding in all its chaos.
        """,

        "para_009.txt":
        """
        The college experience is unique for everyone. For some, it's about learning; for others, it's about friends. However, at the end of the day, what I mean to say is that it's unforgettable.
        """,

        "para_010.txt":
        """
        One of the best ways to travel is by train. The scenic beauty, the gentle rocking of the coaches, and, well, it's just a unique experience. That is to say, it's quite different from flying or driving.
        """
}
