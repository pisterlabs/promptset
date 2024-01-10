import openai
import Credentials
import Custom_Message_protocols as sms
import Main

# client initialization

openai.api_key = Credentials.openai_key()


async def gpt_command(msg):

    # ask initial prompt

    user_response = await sms.ask("Input your prompt now. To exit conversation, say #quit.", msg, 60, "")
    list1 = [{"role": "user", "content": user_response}]

    while True:

        if user_response == "#quit" or user_response == "quit" or user_response == "!quit":
            msg.send_sms("Exited conversation.")
            break

        # Generate a response

        Main.gpt_requests += 1
        completion = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=list1
        )

        gpt_response = str(completion.choices[0]["message"]["content"])
        if gpt_response[:2] == "\n\n":
            # gets rid of redundant \n
            gpt_response = gpt_response[2:]

        # append ChatGPT's response to the conversation

        list1.append({"role": "assistant", "content": gpt_response})

        user_response = await sms.ask(f'ChatGPT: {gpt_response}', msg, 60, "#quit")

        # append user's response to the conversation

        list1.append({"role": "user", "content": user_response})

