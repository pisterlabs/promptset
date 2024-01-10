from openai import InvalidRequestError

import agents
from agents import tools


# デバッグ先の指定
agent = agents.MainAgent(verbose=True)
# agent = tools.ProcedureAgent(verbose=True)
run = agent.run

# run = agents.test
# print(run())

while True:
    message = input(">> ")
    if message == "exit" or message == "終了":
        print("会話履歴を保存しますか。（Y/n）")
        input_message = input(">> ")
        if input_message == "Y" or input_message == "y" or input_message == "yes" or input_message == "Yes":
            try:
                buffer = agent.memory.load_memory_variables({})
                with open("chat_history.txt", mode="w", encoding='utf-8') as f:
                    f.write(f'{buffer["chat_history"]}')
                    print("会話履歴を保存しました。")
            except Exception as e:
                print("err : " + str(e))
        print("終了します。")
        break
    try:
        ai_response = run(message)
        print(f'AI : {ai_response}')
    except InvalidRequestError as error:
        message = error.response.json()["error"]["message"]
        print("Hit error: ", message)
        break
    except Exception as e:
        print("err : " + str(e))
        break
