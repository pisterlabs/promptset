import clueai
import cohere
from cohere import Client

api_key = "xxx"
cohere_key = 'xxx'


def get_qas(doc_txt):
    cl = clueai.Client(api_key=api_key, check_api_key=False)
    predictions = cl.doc_qa_generate(model_name='clueai-large', doc=doc_txt).qas[0].qa_pairs
    # print(type(predictions.qas[0].qa_pairs), predictions.qas[0].qa_pairs)
    qas_ls = []
    for index, qa in enumerate(predictions):
        if index > 0 and predictions[index - 1].answer == predictions[index].answer:
            continue
        qas_dict = {'question': qa.question, 'answer': qa.answer}
        qas_ls.append(qas_dict)
    # print(qas_ls)
    return qas_ls


def get_dialogue(prompt):
    # 需要返回得分的话，指定return_likelihoods="GENERATION"
    prompt = prompt + "回答之前格式化纠正你的答案。"
    cl = clueai.Client(api_key=api_key, check_api_key=True)
    prediction = cl.generate(
        model_name='ChatYuan-large',
        prompt=prompt)
    return prediction.generations[0].text


def get_main_points(text: str, api_key=cohere_key):
    co = cohere.Client(api_key)  # This is your trial API key
    response = co.summarize(
        text=text,
        length='auto',
        format='auto',
        model='summarize-xlarge',
        additional_command='translate your summary into Cinese send me',
        temperature=0.3,
    )
    # print('Summary:', response.summary)
    return response.summary


def get_chat(query):
    client = Client(cohere_key)
    chat = client.chat(
        query=query,
        # session_id="1234",
        model="command-xlarge",
        return_chatlog=True
    )
    return chat.reply


if __name__ == '__main__':
    # _, doc_name, fulltxt = query_elastics_fulltext(key="心脏病")
    # print(fulltxt)
    # print(get_dialogue(prompt="静脉曲张有哪些治疗方法？"))
    #     print(get_main_points(text=""" """))
    print(get_chat(query="你可以帮助我干嘛"))
