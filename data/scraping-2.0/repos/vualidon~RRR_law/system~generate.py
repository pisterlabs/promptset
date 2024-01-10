import openai
from listApi import list_api

def rewrite(question, list_api=list_api, idx=0):
    client = openai.OpenAI(api_key=list_api[idx])
    try:
        response = client.completions.create(
            model="gpt-3.5-turbo-instruct",
            prompt=f"""Think step by step to answer this question, and provide search engine queries for knowledge
            that you need. Split the queries with ’;’ and end the queries with ’**’.  
            Question: Tôi nên học ngành Công nghệ thông tin ở trường Tôn Đức Thắng hay trường đại học Công nghệ thông tin?
            Answer: ngành công nghệ thông tin ở trường Tôn Đức Thắng; ngành công nghệ thông tin ở trường đại học Công nghệ thông tin**
            Question: Em đã có bằng MOS, vậy em có được miễn môn Cơ sở tin học ở trường đại học Tôn Đức Thắng không?
            Answer: môn cơ sở tin học ở trường đại học Tôn Đức Thắng; chứng chỉ MOS và miễn môn tại trường Tôn Đức Thắng; điều kiện miễn môn cơ sở tin học trường đại học Tôn Đức Thắng**
            Question: {question}
            Answer:""",
            temperature=0,
            max_tokens=1000
        )

        return [q.strip() for q in response.choices[0].text.split(";")]
    except Exception as e:
        # print(type())
        print(e.args[0])
        if "exceeded your current quota" in e.args[0]:
            list_api.pop(0)
            rewrite(question, list_api, 0)
        else:
            rewrite(question, list_api, (idx+1)%len(list_api))


def generate_answer(prompt, context, list_api=list_api, idx=0):
    if len(list_api) == 0:
        print("No API key left.")
        return {"error": "No API key left."}
    try:
        client = openai.OpenAI(api_key=list_api[idx])
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": f"""You are a helpful assistant.\
                Your name is LEGALBOT.\
                Your job is to answer questions about law base on your knowledge and context given.\
                You are given a context: {context}"""},
                {"role": "user", "content": f"{prompt}"},
            ],
            temperature=0.2,
        )
        return response.choices[0].message.content
    except Exception as e:
        # print(type())
        print(e.args[0])
        if "exceeded your current quota" in e.args[0]:
            list_api.pop(0)
            generate_answer(prompt, context, list_api, 0)
        else:
            generate_answer(prompt, context, list_api, (idx+1)%len(list_api))