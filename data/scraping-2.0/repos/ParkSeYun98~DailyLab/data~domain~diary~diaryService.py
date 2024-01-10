import openai
import config
from datetime import datetime

openai.api_key = config.OPENAI_API_KEY

def createDiary(param, gpt_model):
    builder = DiaryContentBuilder(param)
    user_info = builder.build()

    todo_list, completed_count = build_todo_list(param["todos"], gpt_model)
    completion_rate = completed_count / len(param["todos"]) if param["todos"] else 0

    emotion_count_str = build_emotion_count_per_hour(param["emotionCountPerHour"])
    content = (
        "당신은 한 사람의 완벽한 하루를 위해 연구하고 분석하는 연구원이야. 주어진 내용을 토대로 관찰일지를 작성해줘"
    )

    user_content = (
        user_info + "\n" + 
        "오늘 할 일\n" + 
        "\n".join(todo_list) + "\n\n" +
        
        "시간대별 감정\n" +
        emotion_count_str + "\n\n" + 

        "오늘 할 일과 시간대별 감정 기반으로 다음의 양식을 반드시 지켜서 3인칭 관찰자 시점의 하루 보고서를 작성해줘. \n" +
        "절대로 나이, 생년월일 등 특징은 보고서에 직접 사용 하지마. \n" +
        "보고서의 모든 문장은 '해요' 문체로 작성해줘. \n" +

        "title: 읽고 싶게 자극적인 제목을 기술합니다.\n" +
        "content: 오늘 할 일과 감정을 기반으로 관찰한 하루를 자세하게 '해요' 문체로 기술합니다. \n" +
        "conclusion: 관찰한 내용을 기반으올 분석 결과 및 느낀점을 자세하게 기술합니다. \n" + 
        "advice: 관찰 결론을 기반으로 관찰 대상에게 발전할 수 있도록 향후 방향성에 대해 자세하게 조언, 추천합니다. \n"
    )

    response = openai.ChatCompletion.create(
        model=gpt_model,
            messages=[
            {"role": "system", "content": content},
            {"role": "user", "content": user_content
            }, 
        ], 

        temperature=1, # 상상력
        max_tokens=1500, # 반환 받을 문장 size => 비용
        top_p=1, # 답변의 무작위성, 낮을 수록 답변이 정확하고, 높을 수록 창의적
        frequency_penalty=0.0, # 특정 단어나 phrase 를 포함하지 않도록. -2~2 까지 조정 가능한테, 2에 가까울수록 penalty 가 커진다
        presence_penalty=0.0, # 반복적이지 않은 텍스트를 생성하도록 유도. 반복되며 penalty 부여되며 2에 가까울수록 penalty 가 커진다
    )
    data = response['choices'][0]['message']['content']

    clean_data = data.replace('\n', ' ')

    sections = ['title:', 'content:', 'conclusion:', 'advice:']
    result_dict = {}
    
    for i, section in enumerate(sections):
        start_idx = clean_data.find(section) + len(section)
        end_idx = clean_data.find(sections[i + 1]) if i + 1 < len(sections) else len(clean_data)
        result_dict[section.strip(':')] = clean_data[start_idx:end_idx].strip()

    
    result_dict['score'] = get_score(completion_rate)
    print(result_dict)
    return result_dict

def get_score(rate):
    if rate == 1:
        score = "A+"
    elif rate >= 0.8:
        score = "A"
    elif rate >= 0.6:
        score = "B"
    elif rate >= 0.4:
        score = "C"
    else:
        score = "D"

    return score

def build_todo_list(todos, gpt_model):
    todo_list = []
    sorted_todos = sorted(todos, key=lambda x: x["date"] if x["date"] is not None else [0])
    
    completed_count = 0 

    for todo in sorted_todos:
        todo_str = todo['task']
        
        if gpt_model == "gpt-3.5-turbo-16k":
            if todo["date"] is None:
                date_status = "(수행 하지 않음)"
            else:
                hour, minute = todo["date"][3], todo["date"][4]
                date_status = f"({hour}시 {minute}분 수행함)"
                completed_count += 1

            todo_str += f" - {date_status}"
            
        todo_list.append(todo_str)

    return todo_list, completed_count

def build_emotion_count_per_hour(emotion_count_per_hour):
    emotion_list = []
    for time_range, emotion in emotion_count_per_hour.items():
        emotion_list.append(f"{time_range}: {emotion}")
    return "\n".join(emotion_list)

class DiaryContentBuilder:
    def __init__(self, param):
        self.param = param
        self.content = "오늘 관찰일지 보고서를 작성할 사람은 다음과 같아. \n"

    def _add_gender(self):
        gender_map = {
            'M': "성별: 남자\n",
            'F': "성별: 여자\n"
        }
        self.content += gender_map.get(self.param.get('gender'), '')

    def _add_birthday(self):
        if self.param.get('birthday'):
            self.content += "생년월일: {0}-{1:02}-{2:02}\n".format(*self.param['birthday'])

    def _add_job(self):
        if self.param.get('job'):
            self.content += "직업: {}\n".format(self.param['job'])

    def _add_goal(self):
        if self.param.get('goal'):
            self.content += "이루고 싶은 목표: {}\n".format(self.param['goal'])

    def _add_religion(self):
        if self.param.get('religion'):
            self.content += "종교: {}\n".format(self.param['religion'])

    def _add_age(self):
        if self.param.get('birthday'):
            current_year = datetime.now().year
            birth_year = self.param['birthday'][0]
            age = current_year - birth_year
            self.content += "나이: {}세\n".format(age)

    def build(self):
        self._add_gender()
        self._add_birthday()
        self._add_age() 
        self._add_job()
        self._add_goal()
        self._add_religion()
        return self.content