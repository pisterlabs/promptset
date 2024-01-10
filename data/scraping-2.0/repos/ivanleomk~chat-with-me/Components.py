import openai
from Prompts import QUESTION_PROMPT
import json

from model import AdequateQuestions, MultiQuestionResponse, QuestionSummary


class UserInsight:
    """
    This is an insight block, we use it to allow users to give some degree of suggestions GIVEN a set of prior questions and responses
    """  # noqa: E501

    def __init__(self, desired_response: str):
        self.desired_response = desired_response
        self.prompt = f"Given the following questions and responses, generate some insights which you can give to the client\n{desired_response}"


class Summarize:
    """
    This is a summarization block

    We use summarization blocks in between ClarifyingQuestions and User Insights to provide some responses and context for the questions that they have given.

    This helps the user to generate better suggestions and insights when we use a UserInsights block down the line
    """  # noqa: E501

    def __init__(self, existing_questions: list[str], prompt: str, context: str):
        self.existing_questions = existing_questions
        self.context = context
        self.prompt = prompt

    def run(self):
        completion = openai.ChatCompletion.create(
            model="gpt-3.5-turbo-0613",
            temperature=0.1,
            stream=False,
            functions=[MultiQuestionResponse.openai_schema],
            function_call={"name": MultiQuestionResponse.openai_schema["name"]},
            messages=[
                {
                    "role": "system",
                    "content": self.prompt,
                },
            ],
            max_tokens=1000,
        )

        print("Generated Responses to questions")
        questions = MultiQuestionResponse.from_response(completion).tasks
        for question in questions:
            print(
                f"\nOriginal Question: {question.question}\nResponse: {question.response}"  # noqa: E501
            )


class ClarifyingQuestionBank:
    """
    This is a question block.

    The goal of a question block is to allow an LLM to prompt a user to generate additional questions until he has generated enough questions about a specific set of points.

    Eg. If we want the user to generate questions about the following points:
    - Train the sales staff to sell
    - Lower Prices

    and we get the following questions
    - Have we eaten some eggs in the morning

    It's going to obviously prompt the user to generate more questions.
    """  # noqa: E501

    def __init__(self, desired_context: list[str], user_prompt: str, rounds: int = 3):
        self.desired_context = desired_context
        joined_context = "-\n".join(desired_context)
        self.prompt = QUESTION_PROMPT.format(joined_context=joined_context)
        self.questions = []
        self.user_prompt = user_prompt
        self.rounds = rounds

    def evaluate_user_questions(self):
        question_list = "-\n".join(self.questions)
        completion = openai.ChatCompletion.create(
            model="gpt-3.5-turbo-0613",
            temperature=0.1,
            stream=False,
            functions=[AdequateQuestions.openai_schema],
            function_call={"name": AdequateQuestions.openai_schema["name"]},
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert summarizer. \
                        \
                        \
                    Write a quick summary of the questions the user asked and some feedback on his questions.",  # noqa: E501
                },
                {
                    "role": "user",
                    "content": f"Here are the questions which the user has generated so far:\n-{question_list}. Determine if he has asked at least one question about each point listed above",
                },
            ],
            max_tokens=1000,
        )
        return json.loads(
            completion["choices"][0]["message"]["function_call"]["arguments"]
        )

    def get_user_questions(self):
        print(
            f"{self.user_prompt}\
          (press q or just hit enter once you've generated enough questions )"
        )
        questions = []
        user_input = input(f"#{len(questions)+1} : ")
        while user_input.strip() != "q" and user_input.strip() != "":
            questions.append(user_input)
            user_input = input(f"#{len(questions)+1} : ")

        return questions

    def summarize_user_questions(self):
        question_list = "-\n".join(self.questions)
        completion = openai.ChatCompletion.create(
            model="gpt-3.5-turbo-0613",
            temperature=0.1,
            stream=False,
            functions=[QuestionSummary.openai_schema],
            function_call={"name": QuestionSummary.openai_schema["name"]},
            messages=[
                {
                    "role": "system",
                    "content": self.prompt,
                },
                {
                    "role": "user",
                    "content": f"""Here are the questions which the user has generated so far:\n-{question_list}\nRespond with the following three points in mind
                    
                    - What are the areas that the user has done well in?
                    - What are the areas that the user can improve on?
                    - What are the next steps that the user can take to improve on the areas that he can improve on?
                    
                    Begin with the sentence.
                    Your initial questions covered the scope of .... 
                    """,
                },
            ],
            max_tokens=1000,
        )
        return json.loads(
            completion["choices"][0]["message"]["function_call"]["arguments"]
        )

    def run(self):
        for i in range(self.rounds):
            questions = self.get_user_questions()
            self.questions.extend(questions)

            print("You've asked the following questions so far:")
            for question in self.questions:
                print(f"- {question}")

            evaluation = self.evaluate_user_questions()

            if evaluation["isSufficient"]:
                print(
                    "Congratulations, you've succesfully asked sufficient questions! Let's take a quick look at how you did."
                )
                print("")
                summary = self.summarize_user_questions()
                print(summary["summary"])
                return self.questions

            else:
                print(
                    "That's not quite right. Let's take a step back and consider some other areas which you might have missed"
                )

        print(
            "Unfortunately, you didn't hit all the points that we were looking for. Here's a quick recap on how you might be able to improve: \n"
        )
        summary = self.summarize_user_questions()
        print(summary["summary"])
        return self.questions
