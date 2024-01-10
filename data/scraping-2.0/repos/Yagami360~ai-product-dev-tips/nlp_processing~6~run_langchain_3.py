import argparse
from langchain.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.chat_models import ChatOpenAI

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--openai_api_key', type=str, default="dummuy")
    parser.add_argument('--model_name', type=str, default="gpt-3.5-turbo")
    args = parser.parse_args()

    # ---------------------------------
    # SystemMessagePromptTemplate オブジェクト作成
    # ---------------------------------
    system_template="あなたは、質問者からの質問を{language}で回答するAIです。"
    system_message_prompt = SystemMessagePromptTemplate.from_template(system_template)

    # ---------------------------------
    # HumanMessagePromptTemplate オブジェクト作成
    # ---------------------------------
    human_template="質問者：{question}"
    human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

    # ---------------------------------
    # ChatPromptTemplate オブジェクト作成
    # ---------------------------------
    chat_prompt = ChatPromptTemplate.from_messages(
        [system_message_prompt, human_message_prompt]
    )

    # ---------------------------------
    # ChatPromptTemplate オブジェクトからプロンプト文生成
    # ---------------------------------
    prompt_message_list = chat_prompt.format_prompt(language="アニメ口調", question="VTuber について教えて").to_messages()
    print(prompt_message_list)

    # ---------------------------------
    # Chat 用モデル定義
    # ---------------------------------
    chat_llm = ChatOpenAI(
        openai_api_key=args.openai_api_key,
        model_name=args.model_name,
        temperature=0.9,    # 大きい値では出現確率が均一化され、より多様な文章が生成される傾向がある。低い値では出現確率の高い単語が優先され、より一定の傾向を持った文章が生成される傾向がある。
    )
    print("chat_llm: ", chat_llm)

    # ---------------------------------
    # Chat 用LLM推論実行
    # ---------------------------------
    try:
        response = chat_llm(prompt_message_list)
        print(f"response: {response}")
    except Exception as e:
        print(f"Excception was occurred | {e}")
        exit(1)

    exit(0)
