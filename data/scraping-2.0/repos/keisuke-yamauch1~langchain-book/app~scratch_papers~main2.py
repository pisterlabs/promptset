from langchain.llms import OpenAI


if __name__ == '__main__':
    llm = OpenAI(model_name="text-davinci-003", temperature=0)

    result = llm("自己紹介してください")
    print(result)
