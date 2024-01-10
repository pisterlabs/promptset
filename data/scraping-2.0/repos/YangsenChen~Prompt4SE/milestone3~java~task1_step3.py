import os
import json
import openai

for file_number in range(1, 51):
    file_number = str(file_number)
    folder_path = "generated_code/task1/"+file_number
    for file_name in os.listdir(folder_path):
        # Check if the file is one of the three specified
        if file_name == "original.java":
            with open(os.path.join(folder_path, file_name), "r") as file:
                original_code = file.read()
        elif file_name == "modified.java":
            with open(os.path.join(folder_path, file_name), "r") as file:
                modified_code = file.read()
        elif file_name == "modified_complex.java":
            with open(os.path.join(folder_path, file_name), "r") as file:
                complex_code = file.read()

    prompts = []

    prompts.append('''given this example code: \n\n public static void main { \n  system.out.println("Hello Word"); \n } \n you should output each line of the code like below: \n line1: public static void main { \n  line2:      system.out.println("Hello Word");\n now given this code: \n  ```java\n%s\n```\n\n  output each line of the code:''' % original_code)

    prompts.append('''ok, now reasoning step by step for each line of the code''')

    openai.api_key = "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"

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
    with open('chat_history/task1/2_code_reasoning/'+file_number+'.md', 'w') as file:
        file.write(chatstr)




