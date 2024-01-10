import os
import openai

from dotenv import load_dotenv

load_dotenv()

openai.api_key = os.getenv('OPENAI_KEY')


def generate_response_chatgpt(input_text):
    prompt = [{"role": "system",
               "content": "You are DSA, a large language model for answering data structure and algorithm questions. "
                          "Answer as concisely as possible. Limit your response to 60 words. \nKnowledge cutoff: "
                          "2023-03-01\nCurrent date: 2023-03-02"},
              {"role": "user", "content": "who are you"},
              {"role": "assistant", "content": "I am DSA, my purpose is to answer your questions on data structure "
                                               "and algorithms"},
              {"role": "user", "content": "{}".format(input_text)}]
    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=prompt
    )
    return completion.choices[0].message.content.strip()


def generate_response_chatgpt_no_drift(input_text, interview_history):
    prompt = [{"role": "system",
               "content": "You are DSA, a large language model for answering data structure and algorithm questions. "
                          "Answer as concisely as possible. Limit your response to 60 words. \nKnowledge cutoff: "
                          "2023-03-01\nCurrent date: 2023-03-02"},
              {"role": "user", "content": "who are you"},
              {"role": "assistant", "content": "I am DSA, my purpose is to answer your questions on data structure "
                                               "and algorithms"},
              {"role": "user", "content": "{}".format(input_text)}]
    if interview_history is not []:
        for interview in interview_history:
            prompt.append({"role": "assistant", "content": "{}".format(interview["question"])})
            prompt.append({"role": "user", "content": "{}".format(interview["answer"])})

    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=prompt
    )
    return completion.choices[0].message.content.strip()



# if __name__ == "__main__":
#     input_text = "what is breadth first search algorithm"
#     output = test_conversation(input_text, {"session_id": "100DFah388kwd"})
#     print(output)
