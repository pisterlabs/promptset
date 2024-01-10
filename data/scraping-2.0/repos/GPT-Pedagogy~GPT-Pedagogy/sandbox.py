import openai
from model.Teacher import Teacher


# main: file-Ol4t15mr9T3Wb7cUl4bEtRDj, ft-zJk1gHZzIgGjnNJjMEs7Zbk8


def fine_tune_request(model: str, tuning_file: str):
    response = openai.FineTune.create(training_file=tuning_file, model=model)
    print("Fine tuning request response:", response)


if __name__ == "__main__":

    # {'q': 'Q: In which conflict did France suffer major military losses during World War One? ', 'type': 'mc', 'a': 0, 'choices': ['The Battle of Verdun', 'The Battle of Waterloo', 'The Battle of the Somme', 'The Battle of Marne']}

    # openai.File.create(file=open("mydata.jsonl", "rb"), purpose='fine-tune')
    # response = openai.Completion.create(model="text-davinci-003", prompt="Say this is a test", temperature=0, max_tokens=7)
    # response = openai.FineTune.list()
    # response = openai.File.create(file=open("data/training.jsonl", "rb"), purpose='fine-tune')
    # response = openai.FineTune.create(training_file="file-u34CD1bxZx1muf8xwtk4KHr9", model="ada") # main: file-Ol4t15mr9T3Wb7cUl4bEtRDj
    # response = openai.FineTune.list_events(id="ft-zJk1gHZzIgGjnNJjMEs7Zbk8")
    # response = openai.Completion.create(model="ada:ft-529-2023-02-19-22-00-52", prompt="oliver", temperature=0, max_tokens=7)
    # response = openai.File.list()
    # print(response)

    teacher = Teacher("000")
    """result = teacher.gen_quiz_questions(
        ["The largest fish", "The skeleton of sharks", "Freshwater fish", "Deep see fish", "The classification of fish"],
        ["mc", "mc", "mc", "sa", "sa"]
    )"""

    result = teacher.evaluate.eval_short_answer("Q: What was a notable achievement of Charles Darwin?",
            "Charles Darwin is most well known for his theory of evolution that he published in his book"
            "On the Origin of Species.")
    print(result)

    # m_resp = teacher.gen_multiple_choice("dogs")
    # s_resp = teacher.gen_short_answer("dogs")
    # print(m_resp)
    # print(s_resp)
