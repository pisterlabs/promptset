"""
Idea:
GPT-4 evaluation has become a common use case as of late, with varying
degrees of success. The idea is to have GPT-4 evaluate the answer given
by the bot and compare it to the best answer. The best answer is the
answer that the bot should have given. The bot's answer is the answer
that the bot actually gave. The bot's answer is compared to the best
answer and given a score from 1-10. 1 being the worst and 10 being the
best. The score is then averaged over all the questions in the question
set. The average score is the score for the bot. The higher the score,
the better the bot is at answering questions.

Implementation:
The implementation of this idea is fairly simple. The GPT-4 is given a
question and the relevant correct info. GPT-4 is then asked to evaluate 
the answer given by the bot on a 1-10 scale. We use the answer_set.json
file generated from run_all_bots.py script as the input.
"""
import os
import openai
import json
from tqdm import tqdm

openai.api_key = os.getenv("OPENAI_API_KEY")

with (open("/home/msaad/workspace/honors-thesis/evaluate/answer_set.json", "r")) as f:
    answer_set = json.load(f)


def eval_model(
        question: str,
        answer: str, 
        eval_info: str
    ):
    """
    Evaluating answers on a 1-10 scale.

    This function evaluates the answer given by the bot and compares it
    to the best answer. Eval info contains the most important information
    the bot needs to answer the question. The bot's answer is the answer 
    that the bot actually gave.
    """
    assert(isinstance(question, str))
    assert(isinstance(answer, str))
    assert(isinstance(eval_info, str))

    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            { "role": "system", "content": "You are an evaluation bot. Given a question, and 'eval_info', which contains the key parts of the question that should be addressed in an answer, provide a rating on a scale of 1-10 for the given answer."},
            { "role": "user", "content": f"The question is: {question}. \nThe answer is: {answer}. \nThe eval_info is: {eval_info}. Think about it to yourself, quietly, before answering the question. It is important that the evaluation score reflects whether an answer sufficiently answered the question. Only respond with a number, do not provide any explanation or additional characters. \nRating: "}
        ],
        temperature=0.1,
        max_tokens=5,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )

    return response['choices'][0]['message']['content']


def main():
    """
    Main function for evaluating the model. 

    This facilitates the evaluation of the model by calling the eval_model
    function on each question in the answer_set. The answer_set is a json
    file that contains the questions and answers for the model to evaluate.
    The answer_set is a dictionary with the following structure:

    {
        "Question": {
            "Model_Type": ["Eval Score", "Answer"],
            "Model_Type": ["Eval Score", "Answer"],
            ...
        },
        "Question": {
            "Model_Type": ["Eval Score", "Answer"],
            "Model_Type": ["Eval Score", "Answer"],
            ...
        },
        ...
    }
    """
    print("Evaluating model...")
    
    return_dict = {}
    for question, answers in answer_set.items():
        eval_answer = answers.pop("Eval_Info")
        type = answers.pop("Type")

        out_dict = {"Type": type}
        print("Question:", question)
        for model_type, answer in tqdm(answers.items()):
            out_dict[model_type] = [eval_model(question, answer, eval_answer), answer]

        return_dict[question] = out_dict


    with (open("/home/msaad/workspace/honors-thesis/evaluate/eval_results.json", "w")) as f:
        json.dump(return_dict, f, indent=4)
    
    print("Evaluating model complete.")


if __name__ == "__main__":
    main()