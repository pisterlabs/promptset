import argparse
from langchain.prompts import PromptTemplate 
from langchain.llms import OpenAI
from pydantic import ValidationError

def generate_commit_message(action_item, diff, llm_model):
    template = """
    以下の要約とdiffを元に，gitのコミットメッセージを生成してください。
    返答内容はコミットメッセージだけとします。
    要約:
    {action_item}

    diff:
    {diff}
    """

    prompt = PromptTemplate(template=template, input_variables=["action_item", "diff"])
    prompt_text = prompt.format(action_item=action_item, diff=diff)

    llm = OpenAI(model_name=llm_model)
    commit_message = llm(prompt_text)
    return commit_message

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--action_item", required=True)
    parser.add_argument("--diff", required=True)
    parser.add_argument("--llm_model", default="text-davinci-003", required=True)
    args = parser.parse_args()

    action_item = args.action_item
    diff = args.diff
    llm_model = args.llm_model

    commit_message = generate_commit_message(action_item, diff, llm_model)
    print(commit_message)