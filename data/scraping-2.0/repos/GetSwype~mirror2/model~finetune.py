import json
from messageparser import MessageParser
from core import Conversation

import openai
import logging

logger.setLevel(logging.INFO)


class Finetune:
    def __init__(self, parser: MessageParser):
        self.parser = parser

    def prepare_dataset(self) -> str:
        messages = self.parser.parse()
        cnv = Conversation()
        # print(str(messages[0]))
        # print(str(messages[1]))
        i = 1
        while i < 100:
            _input = messages[i]
            _output = messages[i + 1]
            i += 2

            prompt = cnv.construct_prompt(_input)
            data = {
                "prompt": prompt,
                "completion": _output.text
            }
            # with open("finetuned_dataset.jsonl", "a") as f:
            #     f.write(json.dumps(data) + "\n")
            cnv.add_message(_input)
            cnv.add_message(_output)
            cnv.save()

    def finetune(self):
        job = openai.FineTune.create(training_file="finetuned_dataset.jsonl")
        print("Begun finetune job: ", job.get("id"))
        return job


if __name__ == "__main__":
    parser = MessageParser("imessage", "chat.txt")
    # x = parser.parse()
    finetune = Finetune(parser)
    finetune.prepare_dataset()
