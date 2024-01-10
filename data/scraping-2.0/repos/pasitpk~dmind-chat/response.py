import openai
from prompts import get_prompt, get_edited_prompt
import time


ref_keys = {'Understanding of Question', 'Understanding of Answer', 'Reason', 'Interpretation Result'}
model = "gpt-3.5-turbo"
max_try = 5
temperature = 0
top_p = 0.05


def gpt_request(model, message):
    messages = [{"role": "user", "content": message}]    
    chat = openai.ChatCompletion.create(
            model=model, 
            messages=messages,
            temperature=temperature,
            top_p=top_p
        )
    reply = chat.choices[0].message.content
    return reply


def request_response(qs_id, question, text):
    try_count = max_try
    output = None
    while try_count > 0:
        try_count-=1
        try:
            message = get_prompt(qs_id, question, text)
            output = gpt_request(model, message)
            break
        except Exception as e:
            print('API Error')
            print(e)
            time.sleep(5)

    if output is None:
        return None
    
    try_count = max_try
    while try_count > 0:
        try_count-=1
        try:            
            output = output.replace("'s "," ")
            eval_result = eval(output)     
            text_keys = set(eval_result.keys())
            assert text_keys == ref_keys
            break
        except Exception as e:
            print(e)
            print(output)
            if try_count == 0:
                print('Max tries reached, UNSUCCESSFUL')
                break
            message = get_edited_prompt(qs_id, question, text, output)
            req_try_count = max_try
            while req_try_count > 0:
                try:
                    output = gpt_request(model,message)
                    break
                except Exception as e:
                    print('API Error')
                    print(e)     
                    time.sleep(5)
                req_try_count -= 1

    if 'yes' in eval_result['Interpretation Result'].lower():
        return 1, eval_result
    return 0, eval_result


class OpenAIResponse:

    def __init__(self, key):
        openai.api_key = key


    def get_response(self, user_states, user_id, text):

        response_text = ""

        if text.lower() == "<end>":        
            response_text = "ขอบคุณที่มาคุยกันนะ"
            if user_id in user_states:
                del user_states[user_id]
            
        if text.lower() == "<2q+>":
            user_states[user_id] = self.gen_state("2q+")
            cur_idx = user_states[user_id]['latest_question_index'] + 1
            response_text = user_states[user_id]['user_questions'][cur_idx]
            user_states[user_id]['latest_question_index'] = cur_idx

        if text.lower() == "<2q+_extra>":
            user_states[user_id] = self.gen_state("2q+_extra")
            cur_idx = user_states[user_id]['latest_question_index'] + 1
            response_text = user_states[user_id]['user_questions'][cur_idx]
            user_states[user_id]['latest_question_index'] = cur_idx

        if len(response_text) == 0:
            
            if user_id not in user_states:
                return 'สวัสดีจ้า ฉันคือผู้ช่วยประเมินอาการซึมเศร้า หากคุณต้องการรับการประเมินก็กดปุ่มที่เมนูได้เลย คุณสามารถตอบด้วยเสียงได้ด้วยนะ'
            
            qs_form = user_states[user_id]['qs_form']
            latest_idx = user_states[user_id]['latest_question_index']
            qs_id = user_states[user_id]['qs_ids'][latest_idx]
            promt_question = user_states[user_id]['prompt_questions'][latest_idx]

            score, reason = self.get_score(qs_id, promt_question, text)
            
            if score is None:
                return "ขออภัย ระบบขัดข้อง กรุณาตอบใหม่อีกครั้ง"
            
            user_states[user_id]['answers'].append(text)
            user_states[user_id]['scores'].append(score)
            user_states[user_id]['reasons'].append(reason)

            cur_idx = latest_idx + 1

            if cur_idx < len(user_states[user_id]['user_questions']):
                response_text = user_states[user_id]['user_questions'][cur_idx]
                user_states[user_id]['latest_question_index'] = cur_idx

            else:
                response_text = self.get_final_response(qs_form, user_states[user_id])
                del user_states[user_id]

        return response_text
    
    def gen_state(self, qs_form):
        if qs_form.lower() == '2q+':
            return {
                'qs_form': '2q+',
                'user_questions': [
                    'คุณมีความรู้สึกหดหู่ เศร้า ท้อแท้ หรือ ไม่สบายใจ ในเรื่องอะไรบ้างไหมคะ',
                    'คุณมีความรู้สึกเบื่อ ไม่อยากพูด ไม่อยากทำอะไร หรือทำอะไรก็ไม่สนุกเพลิดเพลินเหมือนเดิมบ้างไหม พอจะเล่าให้ฟังได้ไหมคะ',
                    'ในช่วงหนึ่งเดือนที่ผ่านมานี้ คุณมีความรู้สึกทุกข์ใจจนไม่อยากมีชีวิตอยู่บ้างไหมคะ'
                    ],
                'prompt_questions': [
                    'คุณมีความรู้สึกหดหู่ เศร้า หมดหวัง ไม่สบายใจ เซ็ง ทุกข์ใจ ท้อแท้ ซึม หงอย เครียด กังวล ในเรื่องอะไรบ้างไหมคะ',
                    'ในช่วงนี้ คุณรู้สึกเบื่อ ไม่มีแรงจูงใจ ไม่อยากพูด ไม่อยากทำอะไร หรือทำอะไรก็ไม่สนุกเพลิดเพลินเหมือนเดิมบ้างไหม พอจะเล่าให้ฟังได้ไหมคะ',
                    'ในช่วงหนึ่งเดือนที่ผ่านมานี้ คุณมีความคิดที่ไม่อยากจะมีชีวิตอยู่ต่อไป หรือ บางครั้งเคยมีความคิดอยากตายขึ้นมา หรือ พยายามทำให้ตัวเองจากไปไหมคะ'
                    ],
                'qs_ids': [
                    'qs-01-02',
                    'qs-03',
                    'qs-07-01',
                ],
                'answers': [],
                'scores': [],
                'reasons': [],
                'latest_question_index': -1,
                'latest_response_timestamp': -1,
            }
        if qs_form.lower() == '2q+_extra':
            return {
                'qs_form': '2q+_extra',
                'user_questions': [
                    'คุณมีความรู้สึกหดหู่ เศร้า ท้อแท้ หรือ ไม่สบายใจ ในเรื่องอะไรบ้างไหมคะ',
                    'คุณมีความรู้สึกเบื่อ ไม่อยากพูด ไม่อยากทำอะไร หรือทำอะไรก็ไม่สนุกเพลิดเพลินเหมือนเดิมบ้างไหม พอจะเล่าให้ฟังได้ไหมคะ',
                    'ในช่วงหนึ่งเดือนที่ผ่านมานี้ คุณมีความรู้สึกทุกข์ใจจนไม่อยากมีชีวิตอยู่บ้างไหมคะ'
                    ],
                'prompt_questions': [
                    'คุณมีความรู้สึกหดหู่ เศร้า หมดหวัง ไม่สบายใจ เซ็ง ทุกข์ใจ ท้อแท้ ซึม หงอย เครียด กังวล ในเรื่องอะไรบ้างไหมคะ',
                    'ในช่วงนี้ คุณรู้สึกเบื่อ ไม่มีแรงจูงใจ ไม่อยากพูด ไม่อยากทำอะไร หรือทำอะไรก็ไม่สนุกเพลิดเพลินเหมือนเดิมบ้างไหม พอจะเล่าให้ฟังได้ไหมคะ',
                    'ในช่วงหนึ่งเดือนที่ผ่านมานี้ คุณมีความคิดที่ไม่อยากจะมีชีวิตอยู่ต่อไป หรือ บางครั้งเคยมีความคิดอยากตายขึ้นมา หรือ พยายามทำให้ตัวเองจากไปไหมคะ'
                    ],
                'qs_ids': [
                    'qs-01-02',
                    'qs-03',
                    'qs-07-01',
                ],
                'answers': [],
                'scores': [],
                'reasons': [],
                'latest_question_index': -1,
                'latest_response_timestamp': -1,
            }
    
    def get_score(self, qs_id, question, text):
        return request_response(qs_id, question, text)
    
    def get_final_response(self, qs_form, user_states):

        answers = user_states['answers']
        scores = user_states['scores']
        reasons = user_states['reasons']

        if qs_form.lower() == '2q+':
            return f"""
สรุปคะแนน 2Q+

1. ไม่สบายใจ เซ็ง ทุกข์ใจ เศร้า ท้อแท้ ซึม หงอย : {scores[0]}

2. เบื่อ ไม่อยากพูดไม่อยากทำอะไร หรือทำอะไรก็ไม่สนุกเพลิดเพลินเหมือนเดิม : {scores[1]}

3. มีความรู้สึกทุกข์ใจจนไม่อยากมีชีวิตอยู่ : {scores[2]}

รวมคะแนน : {sum(scores)}
"""   
        
        if qs_form.lower() == '2q+_extra':
            return f"""
สรุปคะแนน 2Q+

1. ไม่สบายใจ เซ็ง ทุกข์ใจ เศร้า ท้อแท้ ซึม หงอย : {scores[0]}
    
    Text answer : {answers[0]}

    Question : {reasons[0]['Understanding of Question']}

    Answer : {reasons[0]['Understanding of Answer']}

    Reason : {reasons[0]['Reason']}


2. เบื่อ ไม่อยากพูดไม่อยากทำอะไร หรือทำอะไรก็ไม่สนุกเพลิดเพลินเหมือนเดิม : {scores[1]}

    Text answer : {answers[1]}

    Question : {reasons[1]['Understanding of Question']}

    Answer : {reasons[1]['Understanding of Answer']}

    Reason : {reasons[1]['Reason']}

    
3. มีความรู้สึกทุกข์ใจจนไม่อยากมีชีวิตอยู่ : {scores[2]}

    Text answer : {answers[2]}

    Question : {reasons[2]['Understanding of Question']}

    Answer : {reasons[2]['Understanding of Answer']}

    Reason : {reasons[2]['Reason']}

    
รวมคะแนน : {sum(scores)}
"""   