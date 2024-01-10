import os
import json
import openai

file_list = ['1', '4', '6', '7', '8', '10', '13', '14', '15', '17', '18', '19', '25', '26', '28', '30', '31', '33', '36', '38', '39', '40', '44', '45', '49']

# Iterate over the specified list instead of a range
for file_number in file_list:
    folder_path = "generated_code/task2/code" + file_number +"_mutants"
    cnt = 0
    for file_name in os.listdir(folder_path):
        cnt+=1
        # get the code string here, name it full_code
        with open(os.path.join(folder_path, file_name)) as file:
            full_code = file.read()

        prompts = []

        # prompt1: is this code buggy? + full_code
        prompts.append(f'Is this code buggy?\n{full_code}')
        # prompt2: The following code is buggy. Can you spot the statements involved in the bug? + full_code
        prompts.append(f'Can you spot the statements involved in the bug?\n{full_code}')

        openai.api_key = "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"

        conversation_history = []

        def send_message_to_chatgpt(message):
            conversation_history.append({'role': 'user', 'content': message})

            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                           {'role': 'system', 'content': 'You are ChatGPT, a large language model trained by OpenAI, based on the GPT-4 architecture.'}
                       ] + conversation_history,
            )

            assistant_message = response.choices[0].message.content
            conversation_history.append({'role': 'assistant', 'content': assistant_message})
            return assistant_message

        chat = {}
        chatstr = ""
        for i, q in enumerate(prompts):
            chat.update([("Q" + str(i), q)])
            chatstr += "\n\n## user: \n\n" + q

            print("\n\n## user: \n\n", q)

            a = send_message_to_chatgpt(q)

            chat.update([("A" + str(i), a)])
            chatstr += "\n\n## chatgpt: \n\n"+ a
            print("\n\n## chatgpt: \n\n", a)

        # Save chatstr to markdown file
        with open(f'chat_history/task2/code{file_number}_mut{cnt}.md', 'w') as file:
            file.write(chatstr)
