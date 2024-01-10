from dotenv import load_dotenv

load_dotenv()  # take environment variables from .env.

from openai import OpenAI
client = OpenAI()




class GPT4_agent:
    def __init__(self, command_docs):
        self.command_docs = command_docs
        self.history = [
            {"role": "system", "content": f"Instructoin: Accomplish the Task given using the Available Commands and by observing what is presented to you. write the command between [command]. For example, [command]go east[command] \n\nAvailable Commands: \n{command_docs}"},
        ]

    def extract_command(self, text):
        try:
            command_tag = '[command]'
            start = text.index(command_tag) + len(command_tag)
            code = text[start:]
            end = code.index(command_tag)
            code = code[:end]
            return code
        except Exception as e:
            return 'command not found. Please put it between [command]'



    def act(self, observation):
        self.history += [
                {'role': 'user', 'content': observation + '\n\n What will you do next? Give me the next command. '}
            ]
        completion = client.chat.completions.create(
            model="gpt-4",
            messages=self.history
        )
        # print(completion)
        response = completion.choices[0].message
        self.history += [{'role':'assistant', 'content': response.content}]

        return self.extract_command(response.content)
