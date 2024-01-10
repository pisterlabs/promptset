import os
import promptlayer
import openai as openai_orig
import openai_mock

env = os.environ.get("APP_DEV")

if env == "dev":
    openai = openai_mock.MockOpenAI
else:
    promptlayer.api_key = os.environ.get("PROMPTLAYER_API_KEY")
    if promptlayer.api_key is None:
        openai = openai_orig
    else:
        openai = promptlayer.openai
        openai.api_key = os.environ.get("OPENAI_API_KEY")


def extract_messages(st):
    default = [
        {
            "role": "system",
            "content": "You are a helpful assistant. Please use concise language to save bandwidth and token usage. Avoid 'AI language model' disclaimer whenever possible.",
        }
    ]

    conversation = st.session_state.get("conversation", {})
    messages = conversation.get("messages", default)

    return messages


# see sample-stream.json to know how to parse it
def generate_stream(st, holder, user_input):
    model = st.session_state["model"]
    messages = extract_messages(st)
    messages.append({"role": "user", "content": user_input})

    print("openai.ChatCompletion.create with", openai, model, messages)
    completion = openai.ChatCompletion.create(
        model=model, messages=messages, stream=True
    )

    # first chunk should be
    # {
    #     "choices": [
    #     {
    #         "delta": {
    #         "role": "assistant"
    #         },
    #         "finish_reason": null,
    #         "index": 0
    #     }
    #     ],
    #     "created": 1684389483,
    #     "id": "chatcmpl-7HQwF5QPvTrDtYPOvBZbzFfDb9tcI",
    #     "model": "gpt-3.5-turbo-0301",
    #     "object": "chat.completion.chunk"
    # }

    # middle chunks are content:
    with holder.container():
        content = ""
        for chunk in completion:
            delta = chunk["choices"][0].get("delta", {})
            if "content" in delta:
                content += delta["content"]
                holder.info(content, icon="🤖")

    # last chunk should be
    # {
    #     "choices": [
    #     {
    #         "delta": {},
    #         "finish_reason": "stop",
    #         "index": 0
    #     }
    #     ],
    #     "created": 1684389483,
    #     "id": "chatcmpl-7HQwF5QPvTrDtYPOvBZbzFfDb9tcI",
    #     "model": "gpt-3.5-turbo-0301",
    #     "object": "chat.completion.chunk"
    # }

    messages.append({"role": "assistant", "content": content})
    st.session_state["messages"] = messages

    # No usage info in stream mode yet
    # https://community.openai.com/t/usage-info-in-api-responses/18862

    return messages


def generate_conversation_title(messages):
    user_messages = [m["content"] for m in messages if m["role"] == "user"]
    conversation = " ".join(user_messages)

    # Generate a prompt for the model
    prompt = f"""
    Based on the following user chat messages ---:

    ---
    {conversation}
    ---

    A title in 5 words or less, without quotes, for this conversation is: """

    # Use the OpenAI API to generate a response
    response = openai.Completion.create(
        engine="text-davinci-002", prompt=prompt, temperature=0.3, max_tokens=60
    )

    # Extract the generated title
    title = response["choices"][0]["text"].strip()
    # remove surrounding quotes
    title = title.replace('"', "")

    return title
