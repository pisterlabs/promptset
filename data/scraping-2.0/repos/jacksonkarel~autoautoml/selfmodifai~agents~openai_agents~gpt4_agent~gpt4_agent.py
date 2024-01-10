import os
import re
import json
from transformers import pipeline
from openai import OpenAI

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
from selfmodifai.non_bash import format_nbl, detect_non_bash_code
from selfmodifai.agents.openai_agents.helpers import conv_history_to_str, update_messages


class Gpt4Agent:
    def __init__(self, manager_data, messages_path, system_prompt):
        self.manager_data = manager_data
        self.messages_path = messages_path
        self.system_prompt = system_prompt

    def run(self):
        messages_path = self.messages_path

        with open(messages_path) as json_file:
            messages = json.load(json_file)

        while True:
            response = client.chat.completions.create(model="gpt-4", messages=messages)

            print(response["usage"]["total_tokens"])

            if response["usage"]["total_tokens"] > 5900:
                response, messages = self.gpt_complete_summarization(messages, messages_path, self.system_prompt)

            else:
                response_content = response["choices"][0]["message"]["content"]

                messages = update_messages(response_content, "assistant", messages, messages_path)

                # Define the regular expression pattern
                pattern = r"```bash\n(.*?)\n```"

                bash_matches = re.findall(pattern, response_content, re.DOTALL)

                non_bash_languages = detect_non_bash_code(response_content)

                bash_response = "Create bash commands that do that. Give me them one by one."
                if bash_matches:
                    content = ""
                    # matches is now a list of all bash commands in the string
                    for bash_command in bash_matches:
                        if bash_command.startswith("cd "):
                            os.chdir(bash_command[3:])
                            continue

                        content += f"{bash_command}:\n"
                        stream = os.popen(bash_command)

                        content += stream.read()

                    if len(content) > 3900:
                        content = "That file is too long to send to you. I only want to send you 25 lines of code at a time. Write bash commands to extract the contents from it in smaller chunks."

                    elif non_bash_languages:
                        nbl_str = format_nbl(non_bash_languages)

                        content += f"Those are the outputs from those bash commands. Can you write bash commands to implement the {nbl_str} code?"

                    elif not content:
                        content = "Ok, did that"

                elif non_bash_languages:
                    languages = format_nbl(non_bash_languages)
                    content = f"Write bash commands to implement those changes in the {languages} files."

                elif "?" not in response_content:
                    content = bash_response

                elif self.manager_data:
                    classifier = pipeline("zero-shot-classification")
                    labels = [
                        "a suggestion for what to do next",
                        "an inquisitive question",
                        "asking somebody to do something",
                        "informative statements",
                    ]
                    results = classifier(
                        sequences=response_content, candidate_labels=labels, hypothesis_template="This text is {}"
                    )

                    result_label = results["labels"][0]
                    gen_message_content = self.manager_data[result_label]
                    if gen_message_content:
                        content = gen_message_content()
                    else:
                        content = self.gpt4_suggestion(messages)

                else:
                    content = bash_response

                messages = update_messages(content, "user", messages, messages_path)

    def gpt4_suggestion(self, messages):
        full_context = "This is a conversation between you and a language model-powered AI agent:\n"
        full_context = conv_history_to_str(messages, full_context, user_name="you", assistant_name="AI agent")
        full_context = f"{full_context}\n\n Write a message to the agent directing them to do what they are trying to help us do. They will accomplish their task by writing bash commands that our computer will execute."

        mananager_agent_messages = [
            {
                "role": "system",
                "content": self.system_prompt,
            },
            {"role": "user", "content": full_context},
        ]

        manager_response = client.chat.completions.create(model="gpt-4", messages=mananager_agent_messages)

        print(f"Manager: {manager_response}")

        content = manager_response["choices"][0]["message"]["content"]

        return content

    def gpt_complete_summarization(self, messages, messages_path, system_prompt):
        response, messages = self.handle_too_long_context(messages, system_prompt)

        with open(messages_path, "w") as outfile:
            json.dump(messages, outfile)

        return response, messages

    def handle_too_long_context(self, messages, system_prompt):
        print("too long context")
        messages = messages[:-1]
        full_context = "Condense the information from the following past conversation between us. Keep all of the information that is relevant to the task at hand and future important tasks and remove all that is not. Keep information about the locations of newly created files that might be helpful for future tasks. Imagine a future person picking up where you left off based on this summary. Please by fairly detailed. The past conversation:\n"

        full_context = conv_history_to_str(messages, full_context)

        print(full_context)
        system_turn = {
            "role": "system",
            "content": system_prompt,
        }

        less_messages = [system_turn, {"role": "user", "content": full_context}]

        response = client.chat.completions.create(model="gpt-4", messages=less_messages)

        print(response["choices"][0]["message"]["content"])
        print(response["usage"]["total_tokens"])

        less_messages = [system_turn, {"role": "assistant", "content": response["choices"][0]["message"]["content"]}]

        return response, less_messages
