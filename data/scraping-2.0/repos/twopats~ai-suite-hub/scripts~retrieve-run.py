import pprint
import argparse
import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

pp = pprint.PrettyPrinter(indent=4)

parser = argparse.ArgumentParser(description='Retrieve a run from OpenAI')

parser.add_argument('--status', action='store_true', help='retrieve run status only')
parser.add_argument('--proceed', action='store_true', help='proceed with run')
parser.add_argument('--default', action='store_true', help='retrieve and print run')


client = OpenAI()

threadId = open("threads.txt", "r").read()
runId = open("runs.txt", "r").read()

run = client.beta.threads.runs.retrieve(
  thread_id=threadId,
  run_id=runId
)

messages = client.beta.threads.messages.list(
            thread_id=threadId
            )

if parser.parse_args().status:
  pp.pprint(run.status)
elif parser.parse_args().proceed:
  if run.status == "requires_action":
    pp.pprint("run requires action")

    pp.pprint(run)
    tool_call_id = 'call_XvtVqdDSvA1NiWbItJfxTqd2'

    run = client.beta.threads.runs.submit_tool_outputs(
        thread_id=threadId,
        run_id=run.id,
        tool_outputs=[
            {
                "tool_call_id": tool_call_id,
                "output": "{success: 'true'}"
            }
        ]
    )

    pp.pprint(run)
    # try:
    #   for message in run.messages:
    #             pp.pprint(message)
    #             # if "tool_call_id" in message.content:
    #             #     pp.pprint("tool call id found")
    #             #     tool_call_id = message.content.split("tool_call_id: ")[1].split()[0]
    #             #     pp.pprint(tool_call_id)
    #
    #                 # run = client.beta.threads.runs.submit_tool_outputs(
    #                 #   thread_id="thread_abc123",
    #                 #   run_id="run_abc123",
    #                 #   tool_outputs=[
    #                 #     {
    #                 #       "tool_call_id": "call_abc123",
    #                 #       "output": "{success: 'true'}"
    #                 #     }
    #                 #   ]
    #                 # )
    #
    #
    #             break
    # except:
    #   pp.pprint("no tool call id found")
elif parser.parse_args().default:
  print("##default##")
  print(run)
  

else:
 pp.pprint("no action specified") 



