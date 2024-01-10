from langchain.chat_models import ChatOpenAI

# from langchain.schema import AIMessage  # AIMessageは応答のメッセージの型
from langchain.schema import HumanMessage, SystemMessage

from measurer import Measurer

chat = ChatOpenAI()
chat_model_measurer = Measurer("chat_model")


def messages_in_message_out_without_system_message():
    #  AIMessage(content="J'aime programmer.", additional_kwargs={})
    chat_model_measurer.start_time("messages_in_message_out_without_system_message")
    result = chat(
        [
            HumanMessage(
                content=(
                    "Translate this sentence from English to Japanese: "
                    "I love programming."
                ),
            ),
        ]
    )
    chat_model_measurer.end_time("messages_in_message_out_without_system_message")

    print(result)
    print(
        (
            "response time: "
            f"{chat_model_measurer.time_result('messages_in_message_out_without_system_message')}"
        )
    )


def messages_in_message_out_with_system_message():
    messages = [
        SystemMessage(
            content="You are a helpful assistant that translates English to Japanese."
        ),
        HumanMessage(content="I love programming"),
    ]

    chat_model_measurer.start_time("messages_in_message_out_with_system_message")
    result = chat(messages)
    chat_model_measurer.end_time("messages_in_message_out_with_system_message")

    print(result)
    print(
        (
            "response time: "
            f"{chat_model_measurer.time_result('messages_in_message_out_with_system_message')}"
        )
    )


def batch_calls_richer_outputs():
    batch_messages = [
        [
            SystemMessage(
                content="You are a helpful assistant that translates English to Japanese."
            ),
            HumanMessage(
                content="I love programming",
            ),
        ],
        [
            SystemMessage(
                content="You are a helpful assistant that translates English to Japanese."
            ),
            HumanMessage(
                content="I love artificial intelligence.",
            ),
        ],
    ]

    chat_model_measurer.start_time("batch_calls_richer_outputs")
    result = chat.generate(batch_messages)
    chat_model_measurer.end_time("batch_calls_richer_outputs")

    print(result)
    print(
        (
            "response time: "
            f"{chat_model_measurer.time_result('batch_calls_richer_outputs')}"
        )
    )


if __name__ == "__main__":
    messages_in_message_out_without_system_message()
    messages_in_message_out_with_system_message()
    batch_calls_richer_outputs()
    results = chat_model_measurer.all_time_result(
        "messages_in_message_out_without_system_message"
    )
    for name, response_time in results.items():
        print(f"{name}: {response_time}")
