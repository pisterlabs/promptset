import os
from openai import OpenAI
import logging
import json

class GPTClassifier:
    # TODO 使用英文是否可以减少 token？
    system_template = """
        你是一个在北京生活多年的家庭主妇,擅长对账单进行分类,分类的类别包括:{},请使用json格式,只返回类别,json中的key为category
    """
    user_template = """
        账单:'{}',来自'{}',应该属于哪类?如果判断不了类别,请使用类别:'unknown'
    """

    class_list = "'餐饮','日常开支','服装鞋帽','护肤品','水电物业','医疗'"
    class_index = {
        "dining": "餐饮",
        "daily expenses": "日常开支",
        "clothing and footwear": "服装鞋帽",
        "skincare products": "护肤品",
        "utilities and properties": "水电物业",
        "medical": "医疗",
        "unknown": "unknown"
    }
    token_count = 0
    client = None


    def __init__(self, api_key, class_list = ""):
        if len(class_list) > 0:
            self.class_list = class_list
        self.token_count = 0
        self.client = OpenAI(api_key=api_key)

    def call(self, item_name, payee):
        system_content = self.system_template.format(self.class_list)
        user_content = self.user_template.format(item_name, payee)

        response = self.client.chat.completions.create(
            model="gpt-3.5-turbo-1106",
            response_format={ "type": "json_object" },
            messages = [
                {"role": "system", "content": system_content},
                {"role": "user", "content": user_content}
            ]
        )
        self.token_count = self.token_count + response.usage.total_tokens

        finish_reason = response.choices[0].finish_reason
        content = json.loads(response.choices[0].message.content)
        if finish_reason != "stop":
            logging.error("使用 gpt 推理分类失败:", finish_reason)
            return ""

        logging.info("gpt context:{}".format(content))
        return content["category"]

    def get_token_count(self):
        return self.token_count

if __name__ == '__main__':
    from category import expense_category_mapping  # noqa: F403
    classifier = GPTClassifier()

    text = classifier.call("7-ELEVEn北京黄寺大街西侧店消费", "7-11(SEB)")
    print(text)
    if text not in expense_category_mapping:
        print("text not in expense_category_mapping")
    else:
        print("text in expense_category_mapping")
