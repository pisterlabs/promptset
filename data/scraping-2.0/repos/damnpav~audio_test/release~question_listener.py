from langchain.chat_models import ChatOpenAI
from db_fun import initialize_cursor, question_listener, question_results, question_send

cursor, conn = initialize_cursor()
with open('../auth_file.txt', 'r') as keyfile:
    api_key = keyfile.read()
llm = ChatOpenAI(openai_api_key=api_key)

with open('question_form.txt', 'r') as file:
    question_form = file.read()

stop_flag = 0
while stop_flag == 0:
    listen_df = question_listener(cursor, conn)

    if len(listen_df) > 0:
        if len(listen_df) > 1:
            print(f'There are more than 1 line! Take last rows')
            listen_df = listen_df[-1:].copy()
        order_id = listen_df['OrderID'].values[0]
        transcribed_text = listen_df['TranscribeText'].values[0]
        question_query = f'{question_form}: "{transcribed_text}"'
        question_send(question_query, order_id, cursor, conn)
        response = llm.invoke(question_query)
        question_results(response.content, order_id, cursor, conn)
        stop_flag = 1



