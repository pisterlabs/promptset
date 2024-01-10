"""This function is used to call openai api to detect intent of user's input"""

import openai
import json
from config import config
from config.constance import project_stages


openai.api_key = config.OPENAI_API_KEY

dict_of_intents = {
    "get_project_stages": "Hỏi về tình trạng của dự án",
    "get_project_milestones": "Hỏi về milestones (những cột mốc, thời điểm quan trọng) của dự án",
    "get_project_activities": "Hỏi về các hoạt động (activity) của dự án",
    "get_project_issues": "Hỏi về các vấn đề (issue) của dự án",
    "get_project_risks": "Hỏi về các rủi ro (risk) của dự án",
    "get_solved_issues": "Hỏi về các vấn đề đã được giải quyết của dự án",
    "get_bot_info": "Hỏi về các intent có thể hỗ trợ",
    "other": "Các câu hỏi khác không nằm trong các intent trên"
}
dict_of_entities = {
    "project_name": "Tên của dự án",
    "time": "Thời gian của dự án",
    "organization": "Tên của công ty hoặc tổ chức",
    "stage": f"Trạng thái của dự án, thuộc các giá trị: {', '.join(project_stages)}",
}

answer_prompt = """Hãy đưa ra intent và entity trong câu hỏi của người dùng (trong câu hỏi sẽ có trường hợp có nhiều 
hơn 1 intent) , câu trả lời chỉ cần ngắn gọn dưới dạng như sau : {"intents": ["intent1", "intent2"], "entities": {
"time":  ["time1", "time2"], "project_name": ["name1", "name2"]}}"""


def create_bot_prompt():
    prompt = """Intents:\n"""
    for intent in dict_of_intents:
        prompt += f"{intent}: {dict_of_intents[intent]}\n"
    prompt += """Entities:\n"""
    for entity in dict_of_entities:
        prompt += f"{entity}: {dict_of_entities[entity]}\n"
    prompt += answer_prompt
    return prompt


bot_prompt = create_bot_prompt()


class IntentDetector:
    def detect(self, user_input):
        user_prompt = f"Câu hỏi: {user_input}"
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": bot_prompt},
                {"role": "user", "content": user_prompt}
            ])
        result = json.loads(response.choices[0].message.content)
        intents = []
        entities = []
        if "intents" in result:
            intents = result["intents"]
        if "entities" in result:
            entities = result["entities"]
        return intents, entities

