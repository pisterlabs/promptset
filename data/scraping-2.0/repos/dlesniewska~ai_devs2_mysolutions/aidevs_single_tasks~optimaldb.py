import json

import openai
import requests

from helper import Helper

OFFLINE = True

class Optimaldb:

    @staticmethod
    def generate_answer(test_data):
        json_database = None
        result = []

        if OFFLINE:
            result = open("aidevs_single_tasks/data/3friends.txt", "r").read()
        else:
            url = str(test_data.json()["database"])
            print(url)
            openai.api_key = Helper().get_openapi_key()
            json_database = requests.get(url=url).json()

            for person in list(json_database.items()):
                who = person[0]
                info = person[1]
                ai_resp = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system",
                         "content": """Rules:###write short summary , use very short words and sentences, text 350 words max ###
                         Examples:###
                         Tom has blue eyes and brown hair. He is 180 cm tall. He likes to play football. Tom is 20 years old.###
                         is
                         Tom blue eyes, Tom brown hair, Tom 180 cm tall, Tom 20 years old, Tom likes football###
                         ###
                """},
                        {"role": "user",
                         "content": str(info)}
                    ])
                json_result = ai_resp.choices[0].message.content

                result.append(f'###{who}:{json_result}###')
                print("OPEN AI Result", json_result)

        return str(result)


if __name__ == '__main__':
    test_data = Helper.create_simulated_response(#
        b'{"ala":["aaa","bbb"],"database":"https://zadania.aidevs.pl/data/3friends.json"}')

    ans = Optimaldb().generate_answer(test_data)
    print(ans)


#Result: {"code": 0, "msg": "OK", "note": "CORRECT", "questions": "\n1. Jaka jest ulubiona gra Zygfryda?\n2. W jakim sklepie pracuje Stefan?\n3. Jaki jest ulubiony film Zygfryda?\n4. Jaki taniec weselny wybra\u0142 Zygfryd na swoim weselu?\n5. Co studiuje Ania?\n6. Jak nazywa si\u0119\u00a0inspiracja fitness Ani? \n", "answers": "1. Ulubiona gra Zygfryda to \"Terra Mystica\".\n2. Stefan pracuje w sklepie \u017babka.\n3. Ulubiony film Zygfryda to \"Matrix\".\n4. Zygfryd wybra\u0142 tango na taniec \u015blubny.\n5. Ania studiuje prawo.\n6. Inspiracj\u0105 fitness Ani jest Jennifer Lopez."}

