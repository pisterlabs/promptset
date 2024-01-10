import openai
from openhexa.sdk import current_run, parameter, pipeline, workspace


@pipeline("chatgpt", name="chatgpt")
@parameter("question", required=True, type=str, default="Hello, how are you?")
def chatgpt(question):
    openai.api_key = workspace.custom_connection("openai").api_key
    answer = ask(question)
    current_run.log_info("Writing answer to file")
    with open(f"{workspace.files_path}/answer.txt", "w") as f:
        f.write(answer)

    current_run.log_info("Done !")


@chatgpt.task
def ask(question: str):
    current_run.log_info(f"Ask ChatGPT to answer '{question}'")
    # list models
    models = openai.Model.list()
    current_run.log_info(models)

    # create a completion
    completion = openai.Completion.create(model="ada", prompt=question)

    # print the completion
    print(completion.choices[0].text)


if __name__ == "__main__":
    chatgpt()
