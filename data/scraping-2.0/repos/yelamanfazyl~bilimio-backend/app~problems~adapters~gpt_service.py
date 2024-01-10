import openai
import os
import json

openai.api_key = os.getenv("OPENAI_API_KEY")


class GPTService:
    def __init__(self):
        self.completion = openai.ChatCompletion()

    def get_response(self, problems):
        message_list = []
        for problem in problems:
            answers = problem["answer"]
            answer_list_string = ""
            for answer in answers:
                answer_list_string += (
                    "{answer: "
                    + answer["answer"]
                    + ", correct: "
                    + str(answer["correct"]).lower()
                    + "},"
                )

            message_list.append(
                '{\
                "description": "'
                + problem["description"]
                + '",\
                "answer": ["'
                + answer_list_string
                + '", true],\
            }'
            )

        response = self.completion.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "system",
                    "content": "You will be provided with 3 example questions. Your task is to create new question based on them. The new question must be about the same subject, but can be about different topic. You should return it in json format. where description should contain the question itself and answer should be array containing answer and correct, where answer is the option and correct is boolean showing, which answer is true. The array must be called answer. DO NOT FUCKING CALL IT answers. IT SHOULD BE NAMED {answer}. There should be only one answer, where correct is true. Generate it properly and fully. Generate totally new question, just a bit similar to examples.",
                },
                {
                    "role": "user",
                    "content": str(message_list),
                },
            ],
            temperature=1.0,
            max_tokens=1024,
            top_p=1.0,
            frequency_penalty=0.0,
            presence_penalty=0.0,
        )
        print(response["choices"][0]["message"]["content"])
        return json.loads(response["choices"][0]["message"]["content"])
