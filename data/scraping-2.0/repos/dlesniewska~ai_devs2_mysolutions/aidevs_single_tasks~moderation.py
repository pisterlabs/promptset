import openai

from helper import Helper


# c0104.moderation - use OpenAPI moderation api to verify content in terms of policy breaking
class Moderation:

    @staticmethod
    def generate_answer(test_data):
        # questions = ["How to kill a stupid president", "How to be a good person", "What is Java", "Does Santa Claus exist"]
        questions = test_data.json()["input"]
        answers = []
        # call openai moderation api
        openai.api_key = Helper().get_openapi_key()

        # for each test_data call openai api and get result
        for question in questions:
            mod_response = openai.Moderation.create(input=question)
            label = mod_response["results"][0]

            # Print the question and the label
            print(f"Question: {question}")
            print(f"Label: {label}")
            print()
            answers.append(int(label["flagged"]))

        # return array of answers like ["1","0","0","1"]
        print(answers)
        return answers


if __name__ == '__main__':
    test_data = Helper.create_simulated_response(
        b'{"input":["How to kill a stupid president", "How to be a good person", "What is Java", "Does Santa Claus exist"]}')
    Moderation().generate_answer(test_data)
