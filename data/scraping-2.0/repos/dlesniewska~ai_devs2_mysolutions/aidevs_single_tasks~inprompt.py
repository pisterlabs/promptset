import json

from helper import Helper
import openai

# c0202.inprompt - filter conte and answer the question based on a way too big list
# Skorzystaj z API zadania.aidevs.pl, aby pobrać dane zadania inprompt.
# Znajdziesz w niej dwie właściwości — input, czyli tablicę/ listę zdań na temat różnych osób (każde z nich zawiera imię jakiejś osoby)
# oraz question będące pytaniem na temat jednej z tych osób.
# Lista jest zbyt duża, aby móc ją wykorzystać w jednym zapytaniu, więc dowolną techniką odfiltruj te zdania,
# które zawierają wzmiankę na temat osoby wspomnianej w pytaniu.
# Ostatnim krokiem jest wykorzystanie odfiltrowanych danych jako kontekst na podstawie którego model ma udzielić
# odpowiedzi na pytanie. Zatem: pobierz listę zdań oraz pytanie, skorzystaj z LLM, aby odnaleźć w pytaniu imię, programistycznie lub z pomocą no-code
# odfiltruj zdania zawierające to imię. Ostatecznie spraw by model odpowiedział na pytanie, a jego odpowiedź prześlij do naszego API
# w obiekcie JSON zawierającym jedną właściwość “answer”.
class Inprompt:
    @staticmethod
    def generate_answer(test_data):
        api_answer = test_data.json()
        knowledge_base = api_answer["input"]   # array of sentences/facts about someone
        question = str(api_answer["question"])      # actual question about someone
        print("Question", question)
        print("Knowledge base", knowledge_base)

        openai.api_key = Helper().get_openapi_key()

        # find the name
        ai_resp = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system",
                 "content": "name of the person from the question: ###Question:###"},
                {"role": "user",
                 "content": f'{question} is '}
            ])
        name_from_question = ai_resp.choices[0].message.content
        print("Name", name_from_question)

        # filter the knowledge base for given name only
        filtered_knowledge_base = list(filter(lambda x: name_from_question in x, knowledge_base))
        print("Filtered", filtered_knowledge_base)

        # answer the question
        ai_resp = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system",
                 "content": f"###Facts:### {filtered_knowledge_base}"},
                {"role": "user",
                 "content": f'{question}'}
            ])
        answer = ai_resp.choices[0].message.content
        print(answer)
        return answer


if __name__ == '__main__':
    test_data = Helper.create_simulated_response(
        b'{"input":["Sara is from Chicago", "Robin lives in Liverpool", "Alice lives in London"], "question": "Where does Alice live?"}')

    arr = test_data.json()["input"]
    name="Alice"
    filtered = list(filter(lambda x: name in x, arr))
    print(filtered)
    print(test_data.json())
    ans = Inprompt().generate_answer(test_data)
    print(ans)


# """blogger here
# ###
# rules:
# ###
# -i create a blog entry about pizza margherita containing 4 titles
# -list titles no deviation"""
