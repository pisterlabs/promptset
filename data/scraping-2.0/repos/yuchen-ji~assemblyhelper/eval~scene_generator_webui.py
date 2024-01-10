import re
import os
import copy
import openai
import gradio as gr

os.environ["http_proxy"] = "http://127.0.0.1:7890"
os.environ["https_proxy"] = "http://127.0.0.1:7890"
openai.api_base = "https://api.ai-yyds.com/v1"
openai.api_key = os.getenv(
    "OPENAI_KEY", default="sk-GM3AyFSCFHwbJdnC4c1a2637E4Bf4433AcFcAc8c3e976cFe"
)


class CodeGenerator:
    def __init__(
        self,
        role="robot",
        file_path=None,
        preprompt=None,
        model="gpt-3.5-turbo",
        oncecall=False,
    ):
        """
        通过file或者str初始化llm的prompt, file的优先级高
        """

        robot_role = "You are a desktop robotic arm with 6 degrees of freedom and a gripper end effector. You need to understand my actions/language and assist me in completing the assembly of the parts."
        scene_role = "You should act as an scene detector used to detect new scene observations after a desktop robot completes its actions."
        valid_role = "You need to act as a validator and answer the validation questions based on given robot code."

        if role == "robot":
            role = robot_role
        elif role == "scene":
            role = scene_role
        elif role == "valid":
            role = valid_role
        else:
            role = ""

        self.model = model
        self.file_path = file_path
        self.preprompt = preprompt
        self.system = [
            {
                "role": "system",
                "content": role,
            },
        ]
        self.history = copy.deepcopy(self.system)
        if file_path:
            with open(file_path, "r", encoding="utf-8") as file:
                self.preprompt = file.read()

        if oncecall:
            self.get_llm_response(user_input=self.preprompt)

    def get_llm_response(self, user_input=None):
        """
        获取llm的反馈api
        """
        if user_input:
            self.history.append({"role": "user", "content": user_input})

        completion = openai.ChatCompletion.create(
            model=self.model,       # model="gpt-3.5-turbo",
            messages=self.history,  # prompt
            temperature=0.2,        # 0~2, 数字越大越有想象空间, 越小答案越确定
            n=1,                    # 生成的结果数
            # top_p=0.1,            # 结果采样策略，0.1只采样前10%可能性的结果
            # presence_penalty=0,   # 主题的重复度 default 0, between -2 and 2. 控制围绕主题程度，越大越可能谈论新主题。
            # frequency_penalty=0,  # 重复度惩罚因子 default 0, between -2 and 2. 减少重复生成的字。
            # stream=False,
            # logprobs=1,           # Modify the likelihood of specified tokens appearing in the completion.
            # stop="\n"             # 结束字符标记
        )

        answer = completion.choices[0].message.content
        self.history.append({"role": "assistant", "content": answer})
        # print(f"ChatGPT: {answer}")
        return answer

    def clear_history(self):
        """
        clear the history
        """
        print("CLEAR")
        self.history = copy.deepcopy(self.system)
        self.get_llm_response(user_input=self.preprompt)


# codeg = CodeGenerator(role="scene",file_path="src/workspaces/scene_description_prompt.yml", model="gpt-3.5-turbo", oncecall=True)
# codeg = CodeGenerator(role="scene",file_path="src/workspaces/scene_description_prompt.yml", model="gpt-4-0613", oncecall=True)

# codeg = CodeGenerator(role="robot",file_path="eval/experiments/prompts/cot_1shot_comment.yml", model="gpt-3.5-turbo", oncecall=True)
codeg = CodeGenerator(role="robot",file_path="eval/experiments/prompts/cot_3shot_comment_scene.yml", model="gpt-4-0613", oncecall=True)

# codeg = CodeGenerator(role="valid", file_path="eval/prompts/validation_prompt.yml", model="gpt-3.5-turbo", oncecall=True)
# codeg = CodeGenerator(role="valid", file_path="eval/prompts/validation_prompt.yml", model="gpt-4-0613", oncecall=True)

# codeg = CodeGenerator()
 
def write_to_file(question, answer):
    result = ""
    answer = re.sub(r'\n\s*\n', '\n', answer)
    result += question + answer + '\n' + '\n'
    with open("eval/experiments/exp_feedback/gpt4_best_result.yml", "a") as f:
        f.write(result)


def answer(question, history=[]):
    history.append(question)
    message = codeg.get_llm_response(question)
    write_to_file(question, message)
    history.append(message)
    responses = [(u, b) for u, b in zip(history[::2], history[1::2])]
    # print(responses)
    return responses, history


with gr.Blocks() as demo:
    chatbot = gr.Chatbot(elem_id="chatbot", label="Assembly Helper", height=800)
    state = gr.State([])
    with gr.Row():
        txt = gr.Textbox(
            show_label=False,
            placeholder="Please input you action/language instrutions.",
            container=False
        )
    with gr.Column(scale=16, min_width=0):
        clear = gr.Button("Clear")

    txt.submit(answer, [txt, state], [chatbot, state])
    txt.submit(lambda: None, None, txt)
    clear.click(codeg.clear_history)
    clear.click(lambda: [], None, chatbot)
    clear.click(lambda: [], None, state)

demo.launch()
