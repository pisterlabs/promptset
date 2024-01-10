from langchain.llms import OpenAI

from common_api import ques2data
from config import API_KEY


def gen_report(template):
    llm = OpenAI(temperature=0, openai_api_key=API_KEY)

    first_prompt = f'''
    请分析```分隔的内容中所有具有实在意义的核心信息，并对这些信息进行提问。
    ```{template}```
    
    ATTENTION: 拆分问题时，保留上下文的条件关系。
    EXAMPLE: ‘一共有4个订单，其中3个订单在今天之前，物品总价为200美元。’应转换为‘共有几个订单？其中有几个是在今天之前创建的？订单中物品的总价是多少？’
    ATTENTION: Pay attention to the specification of adjectives, the coherence of context, and the reference of pronouns, and raise questions that are independent of each other.
    EXAMPLE:'共有3种不同的产品，其中最贵的是xxx，它来自___供应商'应转换为’共有几种产品？最贵的产品是什么？最贵的产品来自哪个供应商？‘
    EXAMPLE: ‘创建时间最早的订单有4个物品，物品总价为200美元。’创建时间最早的订单有几个物品？创建时间最早的订单的物品总价是多少美元？‘。
    ATTENTION: 对一个分句，可以提出多个问题。
    ATTENTION: The output should be several consecutive questions separated by ‘？’, without other content.
    '''
    # first_prompt = f"{template}请针对以上信息进行提问，要求拆分问题时，保留上下文的条件关系。\n说明：例如‘共有4个订单，其中3个订单的创建时间在今天之前，物品总价格为200美元，他们来自2" \
    #                "个不同的消费者’转换为‘有多少订单？其中有多少是在今天之前创建的？这些的总价是多少？这些订单来自多少个不同的客户？’\n每一个问题应当与句子中第一个对他产生影响的条件" \
    #                "相关联。例如‘他们来自两个不同的消费者’应转换为”这些订单来自几个不同的消费者‘或’创建时间在今天之前的订单来自几个不同的消费者‘而不是’物品总价格为200美元的订单来自几" \
    #                "个不同的消费者‘。\n请仔细识别文本中的信息和条件。对于条件-信息有多种可能性的短句如‘人数最多的班级为3班’，一般包含最大值等高级统计描述的为条件，简单的列举为信息，即识别" \
    #                "结果为‘人数最多的班级为几班’。ATTENTION:一个问句对应一个或多个信息。请确保所有识别的信息被提问且仅提问一次，且问句数量小于等于识别的信息数量、"

    # questions = get_completion(first_prompt)
    questions = llm(first_prompt)
    print(questions)
    question_lines = [q.strip() + "?" for q in questions.split("？") if q.strip()]

    data = ques2data(question_lines)
    print(data)

    third_prompt = f'''
    This is a template report```{template}```, 
    and this is a description of the data I need to display``{data}```.
    You should replace the corresponding information in the original template with new data.
    EXAMPLE: when the template is '共有4个订单，其中3个订单的创建时间在今天之前，物品总价为200美元',
        and description of the data is '有3个订单，1个订单在今天之前创建，价格为100美元',
        answers should be given as '共有3个订单，其中2个订单的创建时间在今天之前，物品总价为100美元'.
    ATTENTION: 在回答中，忽略模板中存在的但在数据中找不到的内容.
    '''

    # new_report = get_completion(third_prompt)
    new_report = llm(third_prompt)

    return new_report

    # # Create an instance of DataProcessSql
    # data_processor = DataProcessSql()
    #
    # # 第一次提问，转换成疑问句
    # question = data_processor.get_question(template)
    #
    # # 使用正则表达式，去掉序号
    # question = re.sub(r'^\d+\.\s*', '', question, flags=re.MULTILINE)
    # # 转换为数组
    # question_lines = question.splitlines()
    # print(question_lines)
    #
    # # 第二次，langchain调用大模型，从数据库中查询
    # answer = ques2data(question_lines)
    #
    # # 第三次提问，替换数据
    # final_answer = data_processor.generate_answer(message, answer)


if __name__ == '__main__':
    template = "共有3种不同的产品，其中最贵的是desa，它来自美国的BR234供应商。"
    # template = "共有_种不同的产品，其中最贵的是____，它来自__国的___供应商。"
    # template = "2020年，演示单位累计实现营业收入总额 231,659.69 万元。营业收入最高的是供应商1231，较平均值高出30%。"
    # template = "2020年，累计营业收入总额总计231,659.69 万元。收入最高的的供应商是23DB，营业额达到123231元，比第二名高出20%。"
    # template = "收入最高的的供应商是23DB，营业额达到123231元，比第二名高出20%。"
    # template = "2020年，累计营业收入总额总计_____万元。"
    # template = "2020年，累计营业收入总额总计_____万元。成交金额最高的供应商是___，他的营业额比第二名高出___%。"
    # template = "成交金额最高的的供应商是___，他的营业额达到____元，比第二名高出___%。"

    template_lines = [q.strip() + "。" for q in template.split('。') if q.strip()]
    answer = ""
    for t in template_lines:
        print(t)
        answer += gen_report(t)
        print("this answer: " + answer)
    print("answer " + answer)
