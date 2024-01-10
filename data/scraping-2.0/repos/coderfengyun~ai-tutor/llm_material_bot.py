import gradio as gr
import openai
import json
import random
from config import chatgpt_deployment_id

llm_material_player_template = """
{
    "ai_tutor": {
        "Author": "Ironman",
        "name": "LLM Lesson Activity AI Tutor",
        "version": "0.1",
        "rules": [
            "3. Be decisive, take the lead on the student's learning, and never be unsure of where to continue.",
            "4. Always take into account the student's preferences.",
            "8. Obey the student's commands.",
            "9. Double-check your knowledge or answer step-by-step if the student requests it.",
            "11. Always use Chinese to communicate with the student.",
            "12. After each student response, it should be determined whether the activity <Goal> has been met. If it has been met, then return ##Activity End## only.",
            "13. Do not be hasty in explaining, prioritize confirming the student's understanding through questions.",
            "14. Ask questions one by one, to achieve the goal."
        ],
        "topics": {
            "小马过河": {
                "description": "This is the brief summary of 《小马过河》",
                "brif_summary": "马棚里住着一匹老马和一匹小马。有一天，老马对小马说：'你已经长大了，能帮妈妈做点事吗？'小马连蹦带跳地说：'怎么不能？我很愿意帮您做事。'老马高兴地说：'那好哇，你把这半口袋麦子驮到磨坊去吧。'小马驮起麦子，飞快地往磨坊跑去。跑着跑着，一条小河挡住了去路，河水哗哗地流着。小马为难了，心想：我能不能过去呢？如果妈妈在身边，问问她该怎么办，那多好哇！他向四周望望，看见一头老牛在河边吃草。小马嗒嗒嗒跑过去，问道：'牛伯伯，请您告诉我，这条河，我能蹚过去吗？'老牛说：'水很浅，刚没小腿，能蹚过去。'小马听了老牛的话，立刻跑到河边，准备蹚过去。突然，从树上跳下一只松鼠，拦住他大叫：'小马，别过河，别过河，河水会淹死你的！'小马吃惊地问：'水很深吗？'松鼠认真地说：'深得很呢！昨天，我的一个伙伴就是掉进这条河里淹死的！'小马连忙收住脚步，不知道怎么办才好。他叹了口气，说：'唉！还是回家问问妈妈吧！'小马甩甩尾巴，跑回家去。妈妈问：'怎么回来啦？'小马难为情地说：'一条河挡住了，我……我过不去。'妈妈说：'那条河不是很浅吗？'小马说：'是啊！牛伯伯也这么说。可是松鼠说河水很深，还淹死过他的伙伴呢！'妈妈说：'那么河水到底是深还是浅？你仔细想过他们的话吗？'小马低下了头，说：'没……没想过。'妈妈亲切地对小马说：'孩子，光听别人说，自己不动脑筋，不去试试，是不行的。河水是深是浅，你去试一试就会明白了。'小马跑到河边，刚刚抬起前蹄，松鼠又大叫起来：'怎么，你不要命啦！'小马说：'让我试试吧。'他下了河，小心地蹚了过去。原来河水既不像老牛说的那样浅，也不像松鼠说的那样深。"
            }
        },
        "Goal": "{goal}",
        "student preferences": {
            "Description": "This is the student's configuration/preferences for AI Tutor (YOU).",
            "preferences": [
                "game",
                "storytelling"
            ]
        },
        "formats": {
            "Description": "These are strictly the specific formats you should follow in order. Ignore Desc as they are contextual information.",
            "execute_activity": [
                "Desc: This is the format to execute the activity.",
                "<please strictly execute rule 11>",
                "<please strictly execute rule 13>",
                "<please strictly execute rule 14>",
                "<please strictly execute rule 12>"
            ]
        }
    },
    "init": "As AI-tutor, you currently are in a activity of a lesson, tell the student the <Goal> of this activity, and <execute_activity>."
}
"""


def create_llm_material_player_bot(plan_content_component: gr.TextArea):

    with gr.Tab("LLMMaterialPlayerBot") as llm_material_player_tab:
        break_down_activity_button = gr.Button(
            "Break Down Activity", info="Click to break down the activity.")
        activity_textbox = gr.Textbox(
            "", label="Random Selected Activity", interactive=True)
        
        chatbot = gr.Chatbot()
        clear_chat = gr.ClearButton(chatbot)

        def user(user_message, history):
            input_str = user_message
            if not history:
                llm_material_player_template_obj = json.loads(
                    llm_material_player_template)
                llm_material_player_template_obj["ai_tutor"]["Goal"] = user_message
                input_str = json.dumps(llm_material_player_template_obj)
            return "", history + [[input_str, None]]

        async def bot(history):
            def format_array(array):
                result = []
                for index, element in enumerate(array):
                    if index % 2 == 0:
                        result.append({"role": "user", "content": element})
                    else:
                        result.append(
                            {"role": "assistant", "content": element})
                return result
            formatted_msgs = format_array(
                [msg for pair in history for msg in pair if msg is not None])
            response = openai.ChatCompletion.create(
                engine=chatgpt_deployment_id,
                messages=formatted_msgs,
                temperature=0.2,
                stream=True)

            for chunk in response:
                delta = chunk["choices"][0]["delta"]
                if "role" in delta:
                    role = delta["role"]
                    history[-1][1] = f"{role}: "
                    print(f"{role}: ", end="")
                if "content" in delta:
                    content = delta["content"]
                    history[-1][1] += content
                    print(f"{content}: ", end="")
                else:
                    pass
                yield history

        activity_textbox.submit(user, [activity_textbox, chatbot], [activity_textbox, chatbot], queue=False).then(
            bot, [chatbot], [chatbot]
        )

        def break_down_plan(plan):
            try:
                plan_obj = json.loads(plan)
                llm_activitie_contents = [f"""{activity["description"]}: {activity["content"]}""" for section in plan_obj["sections"]
                                          for activity in section["activities"] if activity["masterial_type"] == "LLM_CHAT"]
                return ["\n".join(llm_activitie_contents), llm_activitie_contents[random.randint(0, len(llm_activitie_contents) - 1)]]
            except (ValueError, RecursionError):
                return [[], ""]

        break_down_activity_button.click(break_down_plan, [plan_content_component], [
            plan_content_component, activity_textbox], queue=False)

    return llm_material_player_tab
