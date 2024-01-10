from langchain.prompts import PromptTemplate, FewShotPromptTemplate


class TemplateManager:
    def __init__(self):
        self.templates = {
            # 如何？
            0 : PromptTemplate(
                input_variables=["question", "related_str"],
                template="""根据相关信息，专业，准确并简要地回答问题。\n问题是：{question}\n已知信息：\n\n{related_str}\n答案是：\n
                """
            ),
            # 总结
            1 : PromptTemplate(
                input_variables=["question", "answer"],
                template="""请根据问题总结答案。问题是：{question}, 答案是：{answer}
                """
            ),
            # few_shot
            "few_shot" : PromptTemplate(
                input_variables=["question", "related_str", "filtered_str","answer"],
                template="你是一位智能汽车说明的问答助手，现在我们节选到了部分说明书中的信息，可能存在着部分无关的信息，请根据说明书的已知信息，筛选出相关信息，然后完整、准确并简要地回答问题，请你回答最直接的答案，不要回答无关内容，或进行解释和延伸。问题是：{question}\n已知信息：{related_str}\n筛选信息：{filtered_str}\n答案是：{answer}"
            )
        }

        self.examples = [
            {
                "question": "在使用FCTA时需要注意哪些事项？",
                "related_str": "当车速处于4-15km/h范围内，车辆驶出停车位或经过交叉路口时，若前方交叉路口预警系统（FCTA）监测到存在前方侧向碰撞风险，将通过仪表提示信息和蜂鸣声提醒驾驶注意观察前方道路情况，从而降低发生前方侧向碰撞的风险。\n<SEP>警告:\n■FCTA属于驾驶辅助系统，并不能确保在任何情况下均能正常工作；紧急情况时，驾驶员应及时接管车辆。\n■虽然车辆配备有FCTA，但为确保行驶安全，通过交叉路口或其他视线受阻的环境前应环顾四周，确认周围环境无危险。\n■FCTA主要对横穿的轿车/SUV、客车/卡车等车辆类型作用，对三轮车、外表不规则的车辆、行人、骑行者、动物具有识别的局限性而不起作用。\n■对于系统识别的有效目标，根据车辆、场景、路况的不同，FCTA并非总能达到相同的性能水平。",
                "filtered_str": "<SEP>警告:\n■FCTA属于驾驶辅助系统，并不能确保在任何情况下均能正常工作；紧急情况时，驾驶员应及时接管车辆。\n■虽然车辆配备有FCTA，但为确保行驶安全，通过交叉路口或其他视线受阻的环境前应环顾四周，确认周围环境无危险。\n■FCTA主要对横穿的轿车/SUV、客车/卡车等车辆类型作用，对三轮车、外表不规则的车辆、行人、骑行者、动物具有识别的局限性而不起作用。\n■对于系统识别的有效目标，根据车辆、场景、路况的不同，FCTA并非总能达到相同的性能水平。",
                "answer" : "在使用FCTA时，需要注意以下事项：\n\n1. FCTA属于驾驶辅助系统，并不能确保在任何情况下均能正常工作，紧急情况时，驾驶员应及时接管车辆。\n2. 虽然车辆配备有FCTA，但为确保行驶安全，通过交叉路口或其他视线受阻的环境前应环顾四周，确认周围环境无危险。\n3. FCTA主要对横穿的轿车/SUV、客车/卡车等车辆类型作用，对三轮车、外表不规则的车辆、行人、骑行者、动物具有识别的局限性而不起作用。\n4. 对于系统识别的有效目标，根据车辆、场景、路况的不同，FCTA并非总能达到相同的性能水平。"
            },
            {
                "question" : "自动模式下，中央显示屏是如何切换日间和夜间模式的？",
                "related_str" : "01点击设置中央显示屏显示模式（日间模式、夜间模式、自动）。\n<SEP>说明:\n您可以依据个人喜好选择自动模式：□日出到日落：白天显示日间模式，晚上显示夜间模式。\n□自定时段：依据设置的时间段切换显示模式。\n□日夜模式选择自动模式后，中央显示屏会自动切换日间模式或夜间模式。\n调节中央显示屏亮度在中央显示屏中点击-设置-显示，进入显示设置界面。\n01点击设置中央显示屏亮暗模式。\n02滑动滑条调节中央显示屏亮度。您还可以通过以下方式调节中央显示屏亮度：■中央显示屏车辆功能界面，请参见车辆功能界面（页码 266）。", 
                "filtered_str": "<SEP>说明:\n您可以依据个人喜好选择自动模式：□日出到日落：白天显示日间模式，晚上显示夜间模式。\n□自定时段：依据设置的时间段切换显示模式。\n□日夜模式选择自动模式后，中央显示屏会自动切换日间模式或夜间模式。",
                "answer" : "在自动模式下，中央显示屏会根据当前环境自动切换日间和夜间模式。您可以在设置中央显示屏显示模式（日间模式、夜间模式、自动）来根据个人喜好选择模式。"
            },
            {
                "question" : "我应该在哪里添加香氛精油？", 
                "related_str" : "香氛精油添加在香氛胶囊中，香氛胶囊位于手套箱内。\n1打开手套箱。\n2向下取下香氛胶囊。\n01香氛胶囊。\n3将香氛精油均匀地涂抹在香芯棒上（三面都要涂抹），静置一分钟。\n4将香氛胶囊安装回位即可。\n",
                "filtered_str" : "香氛精油添加在香氛胶囊中，香氛胶囊位于手套箱内。",
                "answer" : "在香氛精油内添加香氛精油。"
            }
        ]

    def get_template(self, template_name):
        return self.templates.get(template_name, None)
    
    def get_fewshot_template(self):
        few_shot_prompt = FewShotPromptTemplate(
            examples=self.examples,
            example_prompt=self.templates.get("few_shot", None),
            suffix="根据说明书的已知信息，筛选出相关信息，然后完整、准确并简要地回答问题。问题是：{question}\n已知信息：{related_str}\n",
            input_variables=["question", "related_str"]
        )
        return few_shot_prompt

template_manager = TemplateManager()