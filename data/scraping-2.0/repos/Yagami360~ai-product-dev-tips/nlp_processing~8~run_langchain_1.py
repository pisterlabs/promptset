import argparse
from langchain.memory import ChatMessageHistory
from langchain.schema import messages_to_dict
from langchain.schema import messages_from_dict

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--human_prompt', type=str, default="Hello World!")
    parser.add_argument('--ai_prompt', type=str, default="Hello! How can I assist you today?")
    args = parser.parse_args()

    # ChatMessageHistory オブジェクト（Chat の履歴データ（History）を管理する機能）に HumanMessages や AIMessages 追加
    history = ChatMessageHistory()

    history.add_user_message(args.human_prompt)
    history.add_ai_message(args.ai_prompt)
    print(f'history.messages={history.messages}')

    # 履歴の削除
    history.clear()
    print(f'history.messages={history.messages}')

    exit(0)
