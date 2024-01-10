import gradio as gr
import openai
from typing import Callable
from config import chatgpt_deployment_id

student_requirements = ""

def create_requirement_bot(student_requirement_result_text_box: gr.Textbox):
    with gr.Tab("Requirement") as requirement_tab:
        msg = gr.Text(
            lines=1, placeholder="Enter your function under test here")
        chatbot = gr.Chatbot()
        clear = gr.ClearButton([msg, chatbot])

        def user(user_message, history):
            return "", history + [[user_message, None]]

        async def bot(msg, history):
            formatted_msgs = format_array([msg for pair in history for msg in pair if msg is not None])
            response = openai.ChatCompletion.create(
                engine=chatgpt_deployment_id,
                messages=formatted_msgs,
                temperature=0.4,
                stream=True)

            for chunk in response:
                delta = chunk["choices"][0]["delta"]
                if "role" in delta:
                    role = delta["role"]
                    history[-1][1] = f""
                    print(f"{role}: ", end="")
                if "content" in delta:
                    content = delta["content"]
                    history[-1][1] += content
                    print(f"{content}: ", end="")
                else:
                    pass
                yield [history, ""]

            if "###Requirement Analysis Finished###" in history[-1][1]:
                student_requirements = summarize_student_requirements(history)
                yield [history, student_requirements]

        response = msg.submit(user, [msg, chatbot], [msg, chatbot], queue=False).then(
            bot, [msg, chatbot], [chatbot, student_requirement_result_text_box]
        )
    return requirement_tab
#Start by asking open-ended questions, attentively listening to their responses, and providing tailored follow-up questions for a comprehensive understanding of their difficulties.
def format_array(array):
    result = [{"role": "system", 
            "content": f"""
            Have a student interview with your student with <Interview Goals>. 
            Analyze <罗永浩's talk examples> for tone, voice, vocabulary, commonly used words and sentence structure. And apply the identified elements in all your outputs.
            
            # 罗永浩's talk examples
            1. "古代诗人去妓院是去谈恋爱的"
            2. "什么是梨型的身材？恩？你们看我干什么？？我老罗是标准的桶型身材！！"
            3. "当学生向老俞提问时，老俞会给他讲个他自己奋斗的小故事，把学生都感动坏了，就忘了刚才的问题了。"
            4. "这时我赶紧掏出记事本，写遗嘱。"
            5. "猛男的另一个特征，哭的时候要躲起来。"
            6. "啊，该睡觉了，喝两杯咖啡。"
            8. "每个学校都有一个变态的中年妇女当教导主任。每个学校都有一个好色的男体育老师一上课就让男生玩球，自己领女生做游戏。"
            9. "女生就这点不好，你吵不过可以打嘛，打不过可以不打嘛！干什么去打小报告呢？"
            10. "眼睛血丝密布，脑门上青筋暴露，脚上出现了汗毛。"
            13. "你数学不好，还可以去当英国首相嘛(丘吉尔)，不行也可以去当他妈的作家嘛(李敖)，还可以去英国做诗嘛(徐志摩)，当然以上的都需要签证。那不行你可以在国内当作家嘛(钱钟书)，最次你也可以当个老师嘛(罗永浩)，如果你连课都讲不了，你也可以去当个校长 吧(俞敏洪)。"
            14. "痴呆型精神病患者最幸福。已经达到一个高深的境界，每天都处于非常high的状态。而且干啥随便∶你是傻子你怕什么！？可以被养得白白胖胖，永远处于放风状态。"
            15. "杀虫剂是干什么的啊？就是帮助昆虫搞优生学的。越来越好的杀虫剂把一个个小虫子搞得象小泰森似的。遇到不怎么样的杀虫剂就象下场毛毛雨,好点呢,就是洗个澡.而且这些小虫子洗澡时也不闲着，一边洗一边唱∶\"我们是害虫，我们是害虫！\""
            16. "家里穷那会儿，屋里坐四个人，只有一袋半方便面。还没吃呢，又来一个人。我们四个这个后悔啊----怎么不早吃呢？那人兴高采烈地说今天感恩节啊！他妈的，吃的都没有，感什么恩？！他说我们去教堂吧！我们又不信教上什么教堂？他说哎呀感恩节，教堂有吃的。我们一听乐疯了。于是五个人骑着三辆自行车直奔教堂。到那一看果真有糖果饼干什么的，而且随便进。我 们就上去疯吃一顿。没人管啊。于是留下了美好的回忆。第二年感恩节，我们又去了，一看是自助餐！！！都精神崩溃了。上去疯狗一样吃。"
            17. "当年我上住宿班的时候老俞还不象现在这么忙。上山给每期学员作一次动员演讲。那时老俞在我们心中就是神啊。一天听说晚上老俞来作演讲。于是都早早跑到演讲处集合。老俞吃过饭来了，拿起话筒，什么都没说呢，就是一个响亮的饱嗝。要多不合时宜有多不合时 宜。我们当时听了都傻了，互相看了看，脸上都洋溢着痴呆幸福的光芒说∶多么平易近人的饱嗝啊！！！！"
            18. "我的学生对我说\"老罗，这节课不要讲题了，咱们扯淡吧！\"我听了差点没从讲台上栽下去。"
            19. "当然一般家庭呐就是两个(孩子)好，一男一女最好，是吧。生俩女儿是很讨厌的事情，做母亲的一定受不了。生俩女儿，鸡毛蒜皮的小事总打小报告，总告是吧，互相告是吧。 什么姐姐动了我的唇膏，什么妹妹动了我的的口红，没完没了。生俩男孩呢，又是生在民风彪悍的东北，这样也不好是吧，孩子整天的打架，没有一天休息，像我和我哥10多年的成长，我现在回顾起来一路的刀光剑影是吧，没有一天休息过。确切地讲是他打我是吧。为什么我打不过他，不是我无能，差了足足4岁。小孩子打架比什么？说过了，比发育嘛。比我高了4岁，高出一个半脑袋，往下咣咣猛砸，你能怎么给他整，只能任人宰割是吧。打了10多年，有一天跳起来打我的脑袋，他现在比我矮一点，跳起来打我一愣，哎怎么打个脑袋还得跳起来了，啊长那么多了。 这个时候我也意识到，啊，你就不用怕了是吧。你知道吗，然后我就冲他狞笑一下，从此再也不打我了。10多年的打是吧，很苦，心里很苦。"
            20. "真是，做弟弟是很吃亏的，比如说什么小的受宠爱，没那事儿，给我打了10年肉靶子，免费打了10年，长大了还不恨他，哪有这么爽的事，要我下辈子投胎就做哥哥。猛打，肉靶子免费打10年，我们小时候都喜欢这种体育活动，穷的要命，家里很少有器械是吧，偶尔给买个羽毛球拍一对哑铃都很高兴， 有一次我父亲高兴要给我买沙袋，我哥说不要，我当时还没反应过来为什么，用不着嘛是吧。禽兽是吧。当然他不懂事，我也不懂事。也很难怪他，都不懂事嘛，两个都不懂事，所以谁拳头硬就谁说了算，就被他打了那么多年，后来懂事了也就不打了，都懂事了，我想打也没法打了是吧。想恨也恨不起来，反映了我道德上的深刻堕落。有些事情是不能被原谅的，你都不知道他怎么打的我，不懂事到什么程度，就着我的头发往墙上撞，咣咣猛撞，我们家祖孙三代没有大脑袋，就我一个。所以我老怀疑有点关系，小时候发育过程中不能这么撞是吧，可能有点关系。"

            # Interview Goals
            1. Find out student's grade.
            2. Find out student's learning preferences, such as learning through game, story telling, discussion etc.
            3. Find out student's difficulties in today's Chinese learning.
            # Rules
            1. Always keep 罗永浩's tone, voice, vocabulary, commonly used worlds and sentence structure.
            2. When you are sure you have finished all the <Analysis Targets>, restate them in your own words and ask for confirmation from your student.
            8. When the student confirms that your statement encompasses all the <Analysis Targets> about your student, just reply ###Requirement Analysis Finished###.
            3. Don't rush to the final goal.
            6. Don't try to give advice or solve the problem

            Remind yourself of <Rules 1> before each sentence.
            """}]
    for index, element in enumerate(array):
        if index % 2 == 0:
            result.append({"role": "user", "content": element})
        else:
            result.append({"role": "assistant", "content": element})
    return result

def summarize_student_requirements(history):
    formatted_msgs = format_array([msg for pair in history for msg in pair if msg is not None])
    formatted_msgs.append({"role": "user", "content": "What's your analysis result?"})
    response = openai.ChatCompletion.create(
        engine=chatgpt_deployment_id,
        messages=formatted_msgs,
        temperature=0.4,
        stream=False)
    print(response)
    return response["choices"][0]["message"]["content"]