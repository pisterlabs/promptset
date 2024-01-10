import openai
import json


openai.api_key = "Your API-KEY"


def read_messages_from_jsonl(filename):
    """
    Read messages from a .jsonl file.

    :param filename: Path to the .jsonl file.
    :return: A list of messages.
    """
    messages_list = []
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            entry = json.loads(line)
            messages_list.append(entry["messages"])
    return messages_list


def test_by_case(model_id, message):
    """
    Test the model with a given message.

    :param model_id: ID of the model to test.
    :param message: Message to test the model.
    :return: Response from the model.
    """
    completion = openai.ChatCompletion.create(
        model=model_id,
        messages=message
    )
    return completion.choices[0].message["content"]


def test(model_id, file_name):
    """
    Test the model using the messages from a given file.

    :param model_id: ID of the model to test.
    :param file_name: Path to the .jsonl file containing messages.
    :return: A list of responses from the model.
    """
    result_list = []
    messages_list = read_messages_from_jsonl(file_name)
    for message in messages_list:
        result = test_by_case(model_id, message)
        result_list.append(result)
    return result_list


if __name__ == '__main__':
    # Model ID will be obtained from the e-mail when the training is complete.
    # Alternatively, you can use the following code to get the model_id:
    # result = openai.FineTuningJob.list(limit=10)
    # print(result)

    model_id = ""
    file_name = "dev_data.jsonl"
    results = test(model_id, file_name)
    print(results)

