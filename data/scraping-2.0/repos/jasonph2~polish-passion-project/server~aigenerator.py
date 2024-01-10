from config import OPEN_AI_KEY
from openai import OpenAI 
import string
import random
import ast

def with_turbo(conn):
    try:
        with conn.cursor() as cursor:

            # get all the words the user is familiar with
            sql = "SELECT translated_word FROM db.words WHERE familiarity = 5"
            cursor.execute(sql)
            all = cursor.fetchall()

            translator = str.maketrans('', '', string.punctuation)
            individual_words = [word.translate(translator) for sentence in all for word in sentence['translated_word'].split()]
            individual_words = list(filter(lambda x: x.strip() != '', individual_words))
            random.shuffle(individual_words)
            individual_words = set(individual_words)
            print(individual_words)

            # make an OpenAI request
            client = OpenAI(api_key=OPEN_AI_KEY)
            completion = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You will be given a list of words or phrases in an arbitrary language. Use these words and only these words to formulate sentences, phrases, or questions in this language. Do not give the English translations. Only use the words given. Give your message content with each phrase being seperated by a new-line character"},
                    {"role": "user", "content": str(individual_words)}
                ]
            )
            # print(completion.choices[0].message.content)
            generated_content = completion.choices[0].message.content
            print(generated_content)
            # new_translated_list = generated_content.split("\n")
            # new_translated_list = list(filter(lambda x: x != '', new_translated_list))
            # print(new_translated_list)
            generated_content = generated_content.split("\n")
            print(generated_content)
            return generated_content
            # return ['Teraz jestem gotowy.', 'Lubię uśmiech.', 'Moja siostra nazywa się Lisa.', 'Dzisiaj jestem bardzo podekscytowany.', 'Jeśli potrzebuję pomocy, mogę iść do biblioteki.', 'Chcę iść na imprezę.', 'Moje śniadanie dzisiaj jest bardzo smaczne.', 'Mój stary brat gra w golfa.', 'Idę na mszę.', 'Jason jest studentem.', 'Moja mama robi pyszny obiad.', 'Jak się nazywasz?', 'Czy możesz mi pomóc?', 'Dlaczego jesteś zmęczony?', 'Czy ty jesteś głodny?', 'Czy mogę kupić butelkę wody?', 'Co się dzieje później?', 'Dziękuję za pomoc.', 'Zapomniałem, co chciałem powiedzieć.', 'Czy rodzina jest dla ciebie ważna?', 'Czy masz brata?', 'Potrzebuję kupić makaron.', 'Czytać jest moją ulubioną czynnością.', 'Uczę się angielskiego.', 'Wczoraj byłem w bibliotece.', 'Jak się masz?', 'Czy ty jesteś gotowy na obiad?', 'Czy mogę iść do domu?', 'Kocham cię bardzo.', 'Co jest na stole?', 'Kiedy wracasz do domu?', 'Czy wiesz, gdzie jest mój telefon?', 'Czy mam tylko pięć minut?', 'Rozumiem, że byłeś zajęty.', 'Czy mogę pożyczyć twój długopis?', 'Czy możemy wrócić o osiem?', 'Czy to jest twój telefon?', 'Czy wiesz, gdzie jest moja rodzina?', 'Czy Masz czas na grę w golfa?', 'Czy masz zmęczony po całym dniu?', 'Czy mogę kupić ci piwo?', 'Czy ty mnie znasz?', 'Czy tam jest coś dla mnie w kuchni?', 'Jak wygląda twój tata?', 'Czy masz rodzeństwo?', 'Czy możemy kupić ser?', 'Czy masz coś do powiedzenia?', 'Czy mogę iść na dworzec?', 'Czy mogę iść do stołu?', 'Czy masz czas na naukę dzisiaj?', 'Czy wiesz, gdzie możemy kupić makaron?', 'Czy możemy pójść na zakupy?', 'Czy masz kosz na śmieci?', 'Czy możemy kupić nowy samochód?', 'Czy jesteś gotowy na urodziny?', 'Czy możemy iść na basen?', 'Czy możemy iść na spacer?', 'Czy masz moje zdjęcie?', 'Czy jesteś gotowy na kolację?', 'Czy możemy pić piwo?', 'Czy możemy iść na lunch?', 'Czy możemy iść do parku?', 'Czy możemy iść na obiad?', 'Czy możemy iść na plażę?', 'Czy możemy iść do teatru?', 'Czy to jest bardzo ważne?', 'Czy jesteś gotowy na kino?', 'Czy masz czystą wodę?', 'Czy możemy iść na pizze?', 'Czy mogę iść do twojego domu?', 'Czy masz kod do mojego telefonu?', 'Czy mogę iść na mokro?', 'Czy jesteś zmęczony po całym dniu?', 'Czy masz siedzenie dla mnie?', 'Czy mogę iść na koncert?', 'Czy mogę iść na spacer z tobą?', 'Czy dostanę deser po obiedzie?', 'Czy mogę iść na randkę z tobą?', 'Czy mogę iść na spotkanie?', 'Czy mogę iść na zakupy z tobą?', 'Czy mogę iść na plażę z tobą?', 'Czy masz czas na śniadanie?', 'Czy masz chociaż trochę wódki?', 'Czy mogę iść na cmentarz?', 'Czy mogę iść na zawody?', 'Czy mogę iść na trening?', 'Czy mogę iść do kina?', 'Czy mogę iść na zawody?', 'Czy masz coś do powiedzenia?', 'Czy mogę iść do toalety?', 'Czy mogę iść na basen z tobą?', 'Czy mogę iść na plażę z tobą?', 'Czy mogę iść na spacer z tobą?', 'Czy masz ulubiony deser?', 'Czy masz głowę do meczu?', 'Czy masz czas na pronunciowanie?', 'Czy mogę iść na pewno?', 'Czy mogę iść na studia?', 'Czy mogę iść do sklepu?', 'Czy mogę iść na boisko?', 'Czy mogę iść na przerwę?', 'Czy mogę iść na kawę?', 'Czy mogę iść na kolację?', 'Czy możemy rozmawiać poza tym?', 'Czy możemy iść na spacer?', 'Czy mogę iść na randkę z tobą?', 'Czy mogę iść do domu z tobą?', 'Czy mogę iść na obiad z tobą?', 'Czy mogę iść na drinka z tobą?', 'Czy masz wiadomości o mojej rodzinie?', 'Czy masz jakieś pytania do mnie?', 'Czy możemy się spotkać jutro?', 'Czy mogę pożyczyć twój długopis?', 'Czy możemy być przyjacielami?', 'Czy mogę iść na lunch z tobą?', 'Czy mogę iść na wózek z tobą?', 'Czy można z tobą porozmawiać?', 'Czy mogę iść na powrót?', 'Czy możemy zostać przyjaciółmi?', 'Czy mogę naprawdę iść do domu?']
    except Exception as e:
        print({"message": f"Error: {str(e)}"})
