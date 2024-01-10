import json
import openai

for file_number in range(1, 51):
    file_number = str(file_number)
    with open('dataset/'+file_number+'.json', 'r') as file:
        data = json.load(file)
        original_string = data['original_string']
        func_name = data['func_name'].split('.')[-1]

    # print(original_string)
    # print(func_name)

    prompts = []

    prompts.append(
    '''change the data flow and control flow of the following java code:\n ```java\n%s\n```\n\n''' % original_string)

    prompts.append(
    '''show me a more complex version of the code with modified data and control flow''')


    # Replace "your_api_key_here" with your actual API key
    openai.api_key = "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"

    # Initialize an empty list to store conversation history
    conversation_history = []

    def send_message_to_chatgpt(message):
        # conversation_history.append({'role': 'system', 'content': 'start the chat'})
        conversation_history.append({'role': 'user', 'content': message})

        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                       {'role': 'system', 'content': 'You are ChatGPT, a large language model trained by OpenAI, based on the GPT-4 architecture.'}
                   ] + conversation_history,
            # max_tokens=150,
            # n=1,
            # stop=None,
            # temperature=0.7,
        )

        assistant_message = response.choices[0].message.content
        conversation_history.append({'role': 'assistant', 'content': assistant_message})
        return assistant_message

    # Example usage of the send_message_to_chatgpt function
    # import pyperclip
    chat = {}
    chatstr = ""
    for i,q in enumerate(prompts):

        chat.update([("Q" + str(i), q)])
        chatstr += "\n\n## user: \n\n" + q
        # pyperclip.copy(chatstr)

        print("\n\n## user: \n\n", q)

        a = send_message_to_chatgpt(q)

        chat.update([("A" + str(i), a)])
        chatstr += "\n\n## chatgpt: \n\n"+ a
        print("\n\n## chatgpt: \n\n", a)

    # with open('5_r.json', 'w') as file:
    #     json.dump(chat, file, indent=4)

    #save chatstr to markdown file
    with open('chat_history/task1/1_change_dataflow_controlflow/'+file_number+'.md', 'w') as file:
        file.write(chatstr)




