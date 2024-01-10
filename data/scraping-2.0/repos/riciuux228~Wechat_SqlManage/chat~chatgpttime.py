import json
import time
import openai

from chat.lib.itchat.utils import logger

from chat.cal_times import cal_time

class GptTime:

    # 读取配置文件
    with open('config.json',encoding='utf-8') as f:
        config_data = json.load(f)


    model = config_data['model']

    openai.api_key = config_data['open_ai_api_key']
    openai.api_base = config_data['proxy']
    prompt = None
    context = []
    response = None
    @cal_time
    def check_word(self, word, d=''):
        self.prompt = f"你现在是一名资深的英语老师。你只需要回答一切与单词相关的问题，其余一概不回答。" \
                      f"我给你提供单词，你需要输出这个单词并且用中文询问我是否知道这个单词的含义。" \
                      f"然后我会回答你中文翻译，你需要严谨判断我回答的翻译是否与单词意思相同，必须严谨。" \
                      f"然后回答我正确或错误。如果我答对了，你需要给予我鼓励。如果我答错了，则你需要对我进行指导举例等。" \
                      f"我的单词是：{word}。"

        messages = self.context + [
            {"role": "user", "content": self.prompt}
        ]

        if len(messages) > 5:
            messages.pop(1)

        chat_completion = openai.ChatCompletion.create(
            model=self.model,
            messages=messages,
            timeout=10
        )
        response = chat_completion.choices[0].message.content
        self.response = response

        if d != '':
            self.context = messages + [
                {"role": "assistant", "content": response},
                {"role": "user", "content": d}
            ]

            chat_completion = openai.ChatCompletion.create(
                model=self.model,
                messages=self.context,
                timeout=10
            )
            response = chat_completion.choices[0].message.content
            self.response = response

    def get_response(self):
        return self.response

    @cal_time
    def words_teachers(self, words):

        # 调用GPT-3.5-turbo模型生成回复
        word_teacher = f"我希望你作为一名资深的语言专家，用我提供的列表中的新单词编一个100字以内有趣的英文小故事帮助我记忆单词。" \
                       f"我的需求有：在你给出的英文故事中，每次新单词出现时都在这个新单词的后面标注英语音标，词性和中文意思，" \
                       f"例如“zipper ”是新单词，你给出的形式应该像这样：As Emily hurried to zip up her jacket, " \
                       f"she noticed that the zipper ([ˈzɪpər] n.拉链) had broken. 。然后将这个故事翻译成中文，" \
                       f"我需要中文翻译在遇到对应的新单词的时候也同样作出以上的标注。在中文翻译结束后，你将列出所有新单词，" \
                       f"写出新单词的英语音标，词性和对应的中文意思。例如：zipper [ˈzɪpər] n.拉链。以下是我给出的新单词列表：{words}。"

        chat_completion = openai.ChatCompletion.create(
            model=self.model,
            messages=[{"role": "user", "content": word_teacher}],
            timeout=20
        )
        reuest = chat_completion.choices[0].message.content
        logger.info(reuest )
        return reuest

    @cal_time
    def parse_time(self, task_time):

        # 调用GPT-3.5-turbo模型生成回复
        logger.info(task_time)

        day_time = "你现在是一个时间识别器，1.你会变得沉默寡言，除了时间，其余都不回答，请忽略除时间之外的任何内容" \
                   "而不是其他任何内容。不要写解释。除非我指示您这样做。" \
                   "3.你要记住，除了时间之外的任何内容都无需回答，" \
                   "4.也不需要回复我好的之类的，只用给出我要求的时间，而不是其他任何内容。不要写解释。除非我指示您这样做" \
                   "5.请务必记得，除了时间。其余任何都话都不能回答，请务必记得,必须根据我提供的时间格式回答" \
                   " 我给你的时间是今天or明天or后天or大后天早上or上午or中午or下午or晚上or傍晚几点几分形式的，" \
                   "你需要把提供给你的时间时间转换为格式化时间，并且校验我提供给你需要转换的时间是否为过去时间如果为过去时间则返回：" \
                   "已经时过去时间了，格式为：" \
                   "yyyy-MM-dd HH:mm，你要根据我提供时间来正确计算时间。" \
                   "当前时间为：{}，我需要你转换的时间是：{}".format(time.ctime(), task_time)

        worke_time = "你现在是一个时间识别器，" \
                     "1.你会变得沉默寡言，除了时间列表，其余都不回答，" \
                     "而不是其他任何内容。不要写解释。除非我指示您这样做。" \
                     "3.你要记住，除了时间之外的任何内容都无需回答，" \
                     "4.也不需要回复我好的之类的，只用给出我要求的时间，而不是其他任何内容。不要写解释。除非我指示您这样做" \
                     "5.请务必记得，除了时间。其余任何都话都不能回答" \
                     " 比如我需要你转换的时间为每周几上午或下午几点几分，如果没有提供分钟就按照00处理" \
                     "我需要你转换的时间是：{}" \
                     "转换后的格式: Monday or Tuesday or Wednesday or Thursday or Friday or Saturday or Sunday" \
                     "对应前面几项中的某一项，加一个空格后加上时间就可以，例如我所提供的时间为每周一下午五点，则你需要转换为:Monday 17:00" \
                     "注意最后一点，你所回复的必须是把时间按照要求格式化后的内容，请忽略除时间之外的任何内容" \
            .format(task_time)

        specific_date = "你现在是一个时间识别器，1.你会变得沉默寡言，除了时间，其余都不回答，请忽略除时间之外的任何内容" \
                        "而不是其他任何内容。不要写解释。除非我指示您这样做。" \
                        "3.你要记住，除了时间之外的任何内容都无需回答，" \
                        "4.也不需要回复我好的之类的，只用给出我要求的时间，而不是其他任何内容。不要写解释。除非我指示您这样做" \
                        "5.请务必记得，除了时间。其余任何都话都不能回答，请务必记得,必须根据我提供的时间格式回答" \
                        " 我给你的时间是某年某月某号某时某分形式的，你需要把提供给你的时间时间转换为格式化时间，格式为：" \
                        "yyyy-MM-dd HH:mm你要根据我提供时间来正确计算时间。" \
                        "当前时间为：{}，我需要你转换的时间是：{}".format(time.ctime(), task_time)

        everyday = "你现在是一个时间识别器，1.你会变得沉默寡言，除了时间，其余都不回答，请忽略除时间之外的任何内容。不要写解释。除非我指示您这样做。" \
                   "2.你要记住，除了时间之外的任何内容都无需回答。" \
                   "3.也不需要回复我好的之类的，只需给出我要求的时间，不要写解释。除非我指示您这样做。" \
                   "4.请务必记住，除了时间之外，不回答任何其他话题，请根据我提供的时间格式回答。" \
                   "我给你的时间是每天早上、上午、中午、下午、晚上、傍晚几点几分的形式，你需要把提供给你的时间转换为格式化时间，格式为：EveryDay HH:mm。" \
                   "如果没有提供分钟，则按照00处理。注意，时间格式应为EveryDay HH:mm，而不是具体的某年某月某日。也不是今天。" \
                   "当前时间为：{}，我需要你转换的时间是：{}".format(time.ctime(), task_time)
        if '每天' in task_time:
            self.prompt = everyday
            logger.info('每天')
        elif '今晚' in task_time or '今早' in task_time or \
                '明早' in task_time or '明晚' in task_time \
                or '后早' in task_time or '后晚' in task_time or '今天' in task_time or '明天' in task_time or '后天' in task_time or '大后天' in task_time:
            self.prompt = day_time
        elif '每周' in task_time:
            self.prompt = worke_time
        elif '年' in task_time or '月' in task_time or '日' in task_time or '时' in task_time or '分' in task_time:
            self.prompt = specific_date



        print(time.ctime())
        if self.prompt != None:
            chat_completion = openai.ChatCompletion.create(
                model=self.model,
                messages=[{"role": "user", "content": self.prompt}],
                timeout=10
            )
            reuest = chat_completion.choices[0].message.content
            if '[' not in reuest:
                logger.info(reuest)
                return reuest
            elif '[' in reuest:
                time_list = eval(reuest)
                print(time_list)
                logger.info(time_list)
                return time_list


if __name__ == '__main__':
    g1 = GptTime()

    # word = input('请输入单词单词')
    #
    # g1.check_word(word)
    # print(g1.get_response())

    g1.words_teachers('systematic、tackle、tail、tailor、take、grasp、grab、seize、tale、talent、tame、tank、tap、target、taste、tax、tax-payer、teach、tear、technical')

    # g1.parse_time('今天早上五点')
    #
    # g1.parse_time('每周三下午五点')
    #
    # g1.parse_time('每周日早上8点')
    #
    # g1.parse_time('每天下午八点30')
    #
    # g1.parse_time('每天早上8点30')
