from openai import OpenAI
import json

user_input = input("Enter your API Key: ")
client = OpenAI(api_key=user_input)

file_path = 'train_108900.json'

with open(file_path, 'r', encoding='utf-8') as json_file:
    # Load the JSON data from the file into a Python data structure
    data = json.load(json_file)

custom_output = [{
    'name' : 'convert_answer',
    'description': '產生一個故事，以及他相對應的問題與答案',
    'parameters': {
            'type': 'object',
            'properties': {
                'story': {
                    'type': 'string',
                    'description': '一個2-3句話的故事'
                },
                'question': {
                    'type': 'string',
                    'description': '關於與上故事的問題'
                },
                'answer': {
                    'type': 'string',
                    'description': '關於與上故事的問題的答案，要是文章中部分的文字'
                },
                'answer_start_position': {  
                    'type': 'integer',
                    'description': '答案在故事中的起始位置'
                },
                'question_options': {
                   'type': 'array', 'items': {
                        'type': 'string'
                    }, 'description': "四個選項"
                },
            }
        },
    "required": ["story", "question", "answer", 'answer_start_position', "question_options"],
}]

output = []
error_list = []

for i in range(100):
    try:
        completion = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "你是一位人工智慧助理，請給用戶一個睡前故事，以及他相對應的問題與答案。 以下是幾個例子：{'story': '從前有一隻聰明的狐狸，他總是能找到最好吃的食物。一天，狐狸聽說有一個農場裡有許多雞蛋，於是他決定去偷一些。狐狸找到了農場，看見了一個籠子擺在地上，圍繞著籠子的是一個小溝渠，狐狸知道這是一個詭計。他試著跳過溝渠，但是失敗了。然後，狐狸試著在籠子旁邊挖洞，同樣地，他也失敗了。最後，狐狸想到了一個聰明的辦法，他用木棍把籠子從溝渠中拉了出來。狐狸獲得了雞蛋，並逃走了。', 'question': '狐狸是如何最終逃走的？', 'answer': '他用木棍把籠子從溝渠中拉了出來。', 'answer_start_position': 227} {'story': '從前有一個小男孩，他非常喜歡冰淇淋。每天下午放學後，他都會去附近的冰淇淋店買一球冰淇淋。他最喜歡的口味是巧克力。他覺得這是世界上最好吃的冰淇淋，因為它既甜又濃郁。有一天，他去買冰淇淋時發現店裡沒有巧克力口味的了，他感到非常失望。店主告訴他，巧克力冰淇淋賣完了。小男孩離開店時很難過，但他決定下次再來。他相信，下一次他一定能買到他最愛的巧克力冰淇淋。', 'question': '小男孩最喜歡的冰淇淋口味是什麼？', 'answer': '巧克力', 'answer_start_position': 54}"},
                {"role": "user", "content": f"請給我一個中文故事，以及他相對應的問題與故事中找的到的答案以及。"},
            ],
            functions = custom_output,
            function_call = 'auto'
        )
        json_response = json.loads(completion.choices[0].message.function_call.arguments)
        print(json_response)
        output.append(json_response)
    except:
        print("Error")
        error_list.append(i)
        continue

print(error_list)
with open('output.json', 'w', encoding='utf-8') as f:
    json.dump(output, f, ensure_ascii=False, indent=4)