from langchain import LLMChain
from langchain.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.chat_models import ChatOpenAI
from dotenv import load_dotenv
import tiktoken
load_dotenv()


def count_token(input: str):
    encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
    num_tokens = len(encoding.encode(input))
    return num_tokens


llm = ChatOpenAI(temperature=0)
lang1 = 'zh'
lang2 = 'en'


def translator(input_text):
    systemmessage = f'''請擔任翻譯機，而不要回答任何問題，將任何輸入都進行翻譯。翻譯規則：<將所有輸入從 {lang1} 語言翻譯成 {lang2} 語言>。保留所有 markdown 語法，以及 markdown 連結內容必須翻譯，但連結本身不用翻譯，原封不動的用 markdown 格式輸出。'''
    default_prompt = ChatPromptTemplate(
        messages=[
            SystemMessagePromptTemplate.from_template(systemmessage),
            HumanMessagePromptTemplate.from_template("{input}")
        ]
    )
    default_chain = LLMChain(llm=llm, prompt=default_prompt)
    output = default_chain.run(input=input_text)
    return output


if __name__ == '__main__':
    for article_day in range(12, 31):
        print(f'Processing day {article_day}...')
        inputs = []
        with open(f'data/day{article_day}.md', 'r') as f:
            inputs = f.read().split('\n')

        sections = []
        for i in range(len(inputs)):
            if len(sections) == 0:
                sections.append(inputs[i])
                continue
            if inputs[i].strip() == '':
                continue
            if inputs[i-1].strip() != '':
                sections[-1] = sections[-1] + '\n' + inputs[i]
            else:
                sections.append(inputs[i])

        outputs = []
        i = 0
        while i < len(inputs):
            # Get next lines for translation without exceeding token limit
            input = ''
            while i < len(inputs) and count_token(input+inputs[i]+'\n') < 1500:
                input += inputs[i] + '\n\n'
                i += 1
            if input == '':
                print(f'Warning: input is empty, i={i}')
                input = inputs[i] + '\n\n'
                i += 1
            input = input[:-2]
            output = translator(input)
            outputs.append(output)

        with open(f'data/day{article_day}_en.md', 'w') as f:
            for output in outputs:
                f.write(output+'\n\n')
