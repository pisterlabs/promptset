import subprocess

from openai.types.chat import ChatCompletionToolParam


def make_impl(dir: str, make_args: str, **kwargs):
    # makeコマンドを実行
    results = []
    if make_args.startswith("make "):
        make_args = make_args[5:]
    res_make = subprocess.run(
        ["make"] + make_args.split(" "), cwd=dir, stdout=subprocess.PIPE
    )
    results.append(res_make.stdout.decode("utf-8"))

    return {"message": "\n".join(results), "file": None}


make_tool: ChatCompletionToolParam = {
    "type": "function",
    "function": {
        "name": "make",
        "description": "Run make command by subprocess.",
        "parameters": {
            "type": "object",
            "properties": {
                "dir": {
                    "type": "string",
                    "description": "Directory path to Makefile. e.g. /home/username/projectname/",
                },
                "make_args": {
                    "type": "string",
                    "description": "Args of make command. e.g. test",
                },
            },
            "required": ["dir", "make_args"],
        },
    },
}
