import pyzenbo
import openai
import zhconv
from pyzenbo.modules.dialog_system import RobotFace

#host = '172.20.10.4'
host = '192.168.0.38'

try:
    sdk = pyzenbo.connect(host)
except pyzenbo.PyZenboError as e:
    print(f"連接到 Zenbo 時出錯: {e}")
    exit()

openai.api_key = ''

conversation_history = []

def get_openai_response(prompt):
    prompt_with_history = " ".join(conversation_history + [prompt])
    try:
        response = openai.Completion.create(
            engine='text-davinci-003',
            prompt=(prompt_with_history + "答案不要太冗長"),
            temperature=0.5,
            max_tokens=1024,
            n=1,
            stop=None,
        )
        return response.choices[0].text.strip()
    except openai.error.OpenAIError as e:
        print(f"OpenAI API 錯誤: {e}")
        return "無法生成回應"

def process_user_input(prompt):
    global conversation_history
    conversation_history.append(prompt)
    
    try:
        sdk.robot.set_expression(RobotFace.PLEASED)#, "你剛剛問我，" + prompt)
        response = get_openai_response(prompt)
        reply = zhconv.convert(response, 'zh-hant')
        sdk.robot.speak("我的回答是：" + reply)
        print("我的答案是:" + reply)
        conversation_history.append(reply)
        sdk.robot.speak("你還有什麼想問的嗎？")
    except pyzenbo.PyZenboError as e:
        print(f"與 Zenbo 通信時出錯: {e}")

if __name__ == "__main__":
    sdk.robot.set_expression(RobotFace.PLEASED, "你有什麼想問的嗎？")
    prompt = input("你的問題: ")

    while prompt.lower() != "exit" and prompt.strip() != "":
        process_user_input(prompt)
        prompt = input("你的問題: ")

    sdk.robot.set_expression(RobotFace.HIDEFACE)
    sdk.release()
