from datetime import datetime, timezone

from openai import OpenAI
import json
import logging
from config import config

client = OpenAI(
    api_key=config.openai_api_key,
    organization=config.openai_organization_id
)


def configure_logging(file_path):
    """
    Configures logging settings.
    """
    logging.basicConfig(filename=f"{file_path}/test.log", level=logging.INFO,
                        format="%(asctime)s [%(levelname)s]: %(message)s")
    return logging.getLogger()


def read_messages_from_jsonl(filename):
    """
    Read messages from a .jsonl file.

    :param filename: Path to the .jsonl file.
    :return: A list of messages.
    """
    messages_list = []
    with open(filename, "r", encoding="utf-8") as f:
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
    completion = client.chat.completions.create(
        model=model_id,
        messages=message
    )
    return completion.choices[0].message.content


def test(model_id, file_name):
    """
    Test the model using the messages from a given file.

    :param model_id: ID of the model to test.
    :param file_name: Path to the .jsonl file containing messages.
    """
    messages_list = read_messages_from_jsonl(file_name)
    for message in messages_list:
        system_message = next((msg["content"] for msg in message if msg["role"] == "system"), None)
        user_question = next((msg["content"] for msg in message if msg["role"] == "user"), None)
        result = test_by_case(model_id, message)
        logger.info(f"System >>> {system_message}")
        logger.info(f"User: {user_question}")
        logger.info(f"GPT: {result}")
        logger.info("-------------------------")


if __name__ == "__main__":
    # Configure logger
    log_file_path = input("Please input the test log file path (e.g. 230101_v1): ")
    logger = configure_logging(log_file_path)

    # list all the models fine-tuned.
    result = client.fine_tuning.jobs.list(limit=10)
    result_list = []
    for item in result:
        created_at_utc = datetime.fromtimestamp(item.created_at, tz=timezone.utc)
        created_at_formatted = created_at_utc.strftime('%Y-%m-%d %H:%M:%S')
        fine_tuned_model_info = {
            "model_name": item.fine_tuned_model,
            "created_at": created_at_formatted,
        }
        result_list.append(fine_tuned_model_info)

    for model_info in result_list:
        print(f"Fine-tuning model >>> {model_info}")

    # choose model
    test_model = input("Please input fine_tuning_model_id: ")
    print(f"Model ID >>> {test_model}")
    logger.info(f"Model ID >>> {test_model}")
    logger.info("-------------------------")

    # test the model
    test_type = input("Please input test type (1: test by chat case; 2: test by file): ")
    if test_type == "1":
        print("If you want to exit, please input 'exit'.")
        system_message = input("System >>> ")
        while True:
            user_input = input("User: ")

            if user_input.lower() == 'exit':
                break

            message = [
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_input}
            ]
            test_result = test_by_case(test_model, message)
            print(f"GPT: {test_result}")
            logger.info(f"System >>> {system_message}")
            logger.info(f"User: {user_input}")
            logger.info(f"GPT: {test_result}")
            logger.info("-------------------------")

    else:
        model_id = test_model
        file_name = input("Please input the validation file path (e.g. 230101_v1/test_data.jsonl): ")
        test(model_id, file_name)

    print("Test completed.")
