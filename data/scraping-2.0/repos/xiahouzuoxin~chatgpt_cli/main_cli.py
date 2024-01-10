import os
import argparse
import openai
from prompts import SYSTEM_PROMPT, construct_user_prompt
from knowledge_base import VectorRetrieval

parser = argparse.ArgumentParser()
parser.add_argument('--model', required=False, type=str, default='gpt-3.5-turbo', help='openai model, default gpt-3.5-turbo')
parser.add_argument('--max_tokens', required=False, type=int, default=1024, help='max_tokens, default 1024')
parser.add_argument('--temperature', required=False, type=float, default=0, help='temperature, default 0')
parser.add_argument('--max_history_len', required=False, type=int, default=5, help='max history length, default 5')
parser.add_argument('--knowledge_dir', required=False, type=str, default='./knowledge/docs', help='directory of knowledge files, default ./knowledge/docs')
args = parser.parse_args()

class SimpleChatIO():
    def prompt_for_input(self, role) -> str:
        return input(f">> {role}: ")

    def prompt_for_output(self, role: str):
        print(f">> {role}: ", end="", flush=True)

    def stream_output(self, response):
        collected_messages = []
        for chunk in response:
            chunk_message = chunk['choices'][0]['delta']
            if chunk_message.get('content', None) is not None:
                print(chunk_message['content'], end='', flush=True)
                collected_messages.append(chunk_message['content'])
        print('', flush=True)
        answer = ''.join(collected_messages)
        return answer

def main_cli():
    messaegs = [
        {"role": "system", "content": SYSTEM_PROMPT},
    ]

    if os.path.exists(args.knowledge_dir):
        vector_retrieval = VectorRetrieval('./knowledge/vector_db/')
        n_texts = vector_retrieval.add_index_for_docs(path=args.knowledge_dir)
        if n_texts == 0:
            vector_retrieval = None
            return print('No knowledge files specified.')
        else:
            print('Generate knowledge base success.')
    else:
        vector_retrieval = None

    chatio = SimpleChatIO()
    while True:
        try:
            inp = chatio.prompt_for_input("user")
        except EOFError:
            inp = ""
        if not inp:
            print("exit...")
            break

        content = construct_user_prompt(inp, vector_retrieval)
        messaegs.append({"role": "user", "content": content})
        response = openai.ChatCompletion.create(
                        model=args.model,
                        messages=messaegs[-args.max_history_len:],
                        temperature=args.temperature,
                        max_tokens=args.max_tokens,
                        stream=True,
                    )
        chatio.prompt_for_output('assistant')
        answer = chatio.stream_output(response)
        messaegs.append({"role": "assistant", "content": answer})

main_cli()