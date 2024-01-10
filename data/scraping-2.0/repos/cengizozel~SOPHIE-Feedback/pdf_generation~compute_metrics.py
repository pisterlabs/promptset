import json
import math
import numpy as np
import textstat
import time
import openai

with open('_keys/openai.txt', 'r') as f:
    API_KEY = f.read()
openai.api_key = API_KEY


def get_questions_asked_result(score):
    ranks = {
        (0, 5): 1, # 0 until 20 are included in the interval
        (5, 50): 2
    }

    for interval, rank in ranks.items():
        if interval[0] <= score < interval[1]:
            return rank

    # If score is not in any interval, return "GOOD" because score has to be >= 0
    return 2


def get_open_ended_result(score):
    ranks = {
        (0, 3): 1,
        (3, 50): 2
    }

    for interval, rank in ranks.items():
        if interval[0] <= score < interval[1]:
            return rank

    return 2


def get_hedge_result(score):
    ranks = {
        (0, 15): 1,
        (15, 30): 2
    }

    for interval, rank in ranks.items():
        if interval[0] <= score < interval[1]:
            return rank

    return 2


def get_speaking_rate_result(score):
    ranks = {
        (0, 3.67): 1,
        (3.67, 10): 2
    }

    for interval, rank in ranks.items():
        if interval[0] <= score < interval[1]:
            return rank

    return 2


def get_reading_level_result(score):
    ranks = {
        (0, 13): 1,
        (13, 20): 2
    }

    for interval, rank in ranks.items():
        if interval[0] <= score < interval[1]:
            return rank

    return 2


def get_questions_asked(clinician_lines):
    count = sum(line.count("?") for line in clinician_lines)

    # print all the questions asked
    # for line in clinician_lines:
    #     if "?" in line:
    #         print(line)

    # print("Questions Asked: " + str(count))

    return count

def get_questions(clinician_lines):
    questions = []

    for line in clinician_lines:
        if "?" in line:
            questions.append(line)

    return questions


def get_open_ended(clinician_lines):
    prompt_background = """
    Open-ended questions and closed-ended questions are two types of questions used in communication, research, and survey methods.

    Open-ended questions are questions that do not have a predetermined set of answers, and they encourage the respondent to share their thoughts, feelings, and experiences in their own words. Open-ended questions usually start with words like "What," "How," "Why," or "Tell me about..." and allow for a free-flowing, exploratory answer. These types of questions are useful for gathering qualitative information and for gaining deeper insights into the respondent's perspective or experience.

    Closed-ended questions, on the other hand, are questions that have a limited set of possible answers. They are often used to gather specific information, to confirm details, or to quantify opinions. Closed-ended questions typically start with words like "Is," "Are," "Do," "Did," etc., and elicit either a "yes" or "no" response, or a specific set of choices. These types of questions are useful for gathering quantitative data, but they may not provide as much depth or detail as open-ended questions.

    Open-ended question examples:
        What do you think about the current state of the world?
        How did that experience make you feel?
        Can you tell me about a time when you faced a challenge and how you overcame it?
        What are your thoughts on the recent political developments?
        Can you describe a perfect day in your life?
        What do you think is the most important issue facing our society today?
        How do you prioritize your responsibilities in your daily life?
        Can you share your opinion on the latest technological advancements?
        What do you think makes a successful life?
        How do you approach problem-solving in your work or personal life?

    Closed-ended question examples:
        Did you watch the news yesterday?
        Are you happy with your current job?
        Have you visited a foreign country before?
        Do you prefer sweet or savory foods?
        Is your favorite color blue or red?
        Did you enjoy your last vacation?
        Do you exercise regularly?
        Do you prefer summer or winter?
        Have you read any good books lately?
        Do you believe in ghosts?


    Based on the description and examples above, is the following question open-ended or closed?
    Answer with only "open" or "closed".
    """

    questions_list = []
    for line in clinician_lines:
        if "?" in line:
            questions_list.append(line)

    open_ended_count = 0

    num_tries = 0
    for ql in questions_list:
        num_tries = 0
        limit = 10
        while (num_tries < limit):
            num_tries += 1
            try:
                # print(f"\nQuestion: {ql.rstrip()}")
                response = openai.Completion.create(
                    model="text-davinci-003",
                    prompt=prompt_background + ql.rstrip(),
                    temperature=0.73,
                    max_tokens=256,
                    top_p=1,
                    frequency_penalty=0,
                    presence_penalty=0
                )
                break
            except openai.error.RateLimitError:
                print(
                    f"({num_tries}/{limit}) Rate limit error. Trying again in 2 seconds...")
                time.sleep(2)
        answer = response["choices"][0]["text"].strip()
        # print("Answer: " + answer)
        if answer.lower() == "open":
            open_ended_count += 1

    return open_ended_count


def get_turn_taking(times_file):
    # All endings of each turn
    end_times = []

    # sum time
    total_time = 0 # in seconds
    with open(times_file, "r") as file:
        times_list = file.readlines()
        for timing in times_list:
            timing_info = timing.split(", ")[-1].replace(")", "").strip()
            # print(f"Adding {timing_info} to total time")
            total_time += int(float(timing_info))-1.875# rounded downm to account for transition time
            end_times.append(total_time)

    return end_times


def get_hedge(clinician_lines, hedges_file):
    # Read the hedge words file into a list
    with open(hedges_file, "r") as file:
        hedges = [line.strip() for line in file if line.strip()
                  and not line.startswith("%")]

    # Sort the list of hedge words by length in decreasing order
    hedges = sorted(hedges, key=lambda x: len(x), reverse=True)

    # Create a dictionary to store the count of each hedge word
    hedges_count = {}
    for hedge in hedges:
        hedges_count[hedge] = 0

    # For each line, join all the words into a single string and check if it contains any of the phrases
    word_count = 0
    for line in clinician_lines:
        line = line.split(":")[1].lower().strip()
        word_count += len(line.split())
        for hedge in hedges:
            if hedge in line:
                hedges_count[hedge] += 1
                line = line.replace(hedge, "")

    # Sort the hedge words by their count
    sorted_hedges = sorted(hedges_count.items(),
                           key=lambda x: x[1], reverse=True)

    # # Print the sorted hedge words and their count
    # for hedge, count in sorted_hedges:
    #     if count > 0:
    #         print(hedge, count, end=" | ")
    # print()

    found_hedges = []
    # Add the hedge words the number of times they appear in the dialogue
    for hedge, count in sorted_hedges:
        for i in range(count):
            found_hedges.append(hedge)

    hedge_count = len(found_hedges)

    # print(hedge_count, word_count)
    return found_hedges, hedge_count, word_count

# speaking rate
def get_speaking_rate(clincian_lines, times_file):
    # sum words
    word_count = 0
    for line in clincian_lines:
        line = line.split(":")[1].strip()
        # print(line)
        word_count += len(line.split())
    
    # sum time
    total_time = 0 # in seconds
    with open(times_file, "r") as file:
        times_list = file.readlines()
        for timing in times_list:
            if not timing.startswith("Eta"):
                timing_info = timing.split(", ")[-1].replace(")", "").strip()
                total_time += int(float(timing_info))-1.875# rounded downm to account for transition time
    
    # print(word_count)
    # print(total_time)

    words_per_second = round(word_count / total_time, 2)

    # print(words_per_second)

    return words_per_second


def get_reading_level(clinician_lines):
    grade_level = textstat.flesch_kincaid_grade(" ".join(clinician_lines))

    grade_mapping = {
        0: "Kindergarten",
        1: "1st grade",
        2: "2nd grade",
        3: "3rd grade",
        4: "4th grade",
        5: "5th grade",
        6: "6th grade",
        7: "7th grade",
        8: "8th grade",
        9: "9th grade",
        10: "10th grade",
        11: "11th grade",
        12: "12th grade",
        13: "college graduate"
    }

    reading_level = grade_mapping.get(math.ceil(grade_level), "College Graduate")

    # print("Grade Level: " + str(grade_level))
    # print("Reading Level: " + reading_level)

    return math.ceil(grade_level), reading_level


def main():
    times_file = "E:/SOPHIE/end-of-life-chat-gpt3/times.txt" # only for speaking rate
    # dialogue_file = "docs/conversation-log-obligations/text_processed.txt"
    dialogue_file = "E:/SOPHIE/eta-py/io/sophie-gpt/doctor/conversation-log/text_processed.txt"
    dialogue_lines = []
    with open(dialogue_file, "r") as f:
        dialogue_lines = f.readlines()
    clinician_lines = []
    with open(dialogue_file, "r") as f:
        for line in f.readlines():
            if not line.startswith("Sophie Hallman"):
                clinician_lines.append(line)
    hedges_file = "docs/feedback/hedges.txt"

    hedge_word_cloud, hedge_count, word_count = get_hedge(
        clinician_lines, hedges_file)

    data = {
        "empower": {
            "questions_asked": get_questions_asked(clinician_lines),
            "questions": get_questions(clinician_lines),
            "open_ended": get_open_ended(clinician_lines),
            "turn_taking": get_turn_taking(times_file)
        },
        "explicit": {
            "hedge_word_cloud": hedge_word_cloud,
            "hedge_words": (hedge_count, word_count),
            "speaking_rate": get_speaking_rate(clinician_lines, times_file),
            "reading_level": get_reading_level(clinician_lines)
        },
        "empathize": {
        }
    }

    # print(json.dumps(data, indent=4))

    return json.dumps(data, indent=4)



if __name__ == "__main__":
    main()
