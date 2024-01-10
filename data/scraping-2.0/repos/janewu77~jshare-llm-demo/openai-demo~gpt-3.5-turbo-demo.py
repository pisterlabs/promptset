import cfg.cfg
import openai


# gpt35_turbo的第一个简单demo
# 注意:使用 gpt-3.5-turbo 模型需要 OpenAI Python v0.27.0版本
# 功能：将一段含有日常开销的文字，整理成清晰的收支明细。
def demo_gpt35_turbo_chatcompletion():
    # 新接口增加了role的概念，可以把prompt中的指令、提示等与当前要处理的内容按角色分开。
    # 在这个例子里，我使用了二个role， 分别是system 和 user。
    # System 扮演的角色是讲述要求与规则。
    # User 扮演用户给出一段需要模型处理的文字。

    # “System”，与3.0里的prompt起的作用其实是一样的。
    # 简单来说，就是在AI为你工作前，先唐僧式地给它叨叨上一顿，让它接下来能按照你的要求和规则来产生回复。
    # 这里我给出了一些记帐的规则，要求模型分别记录下支出/收入的明细信息。
    system_prompt = '''
    You are ChatGPT, a large language model trained by OpenAI. Answer as concisely as possible. 
    请记录支出与收入信息。
    当有支出信息时，请记录支出明细（事项、单价、数量、金额)和总支出。
    当有收入信息时，请记录收入明细（事项、金额)和总收入。
    '''

    # 这是一段随手编的文字。模拟记录了多笔琐碎的收入与支出。
    # 有二点可以注意一下：1.文字里故意包含了一些语病。2.里面还有需要计算的金额。
    # 模型将按照“system”的规则与要求，将这段文字进行重新组织后输出。
    input_content = '''
    今天是2023-3-4.
    刚才买了一杯3元的咖啡，买酸奶花了5元，还买了2斤, 15元1斤的小桔子，和朋友一起吃饭又花了300.13元。
    酸奶是直接付的现金，其他是用花呗支付的。
    早上小陈还把上周的我垫付的外卖的钱给了我，一共8元。上午卖苹果收款2028元。昨天收到工资821元。
    '''

    # 重点重点重点！！！ 调用【gpt-3.5-turbo】模型！！！
    res = openai.ChatCompletion.create(
        temperature=0,
        model="gpt-3.5-turbo",  # 指定模型
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": input_content}
        ]
    )
    print(res)

    reply = res['choices'][0]['message']['content']
    # reply = reply.encode('utf-8').decode('unicode_escape')
    print(reply)


if __name__ == '__main__':
    demo_gpt35_turbo_chatcompletion()
