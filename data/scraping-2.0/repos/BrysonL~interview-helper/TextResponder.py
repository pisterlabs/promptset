import openai

# Call the OpenAI Completions API to generate a response to a given message
class TextResponder:
    # Set up the API key and model for use when generating responses
    def __init__(self, api_key, model, starting_messages):
        openai.api_key = api_key
        self.model = model
        self.messages = starting_messages

    # Generate a full response to the given message
    # This returns the full response as a string, but requires the whole string to be generated (by OpenAI) before returning
    def generate_response_full(self, next_message):
        # save the user message so the API has context on the question and past responses
        self.messages.append({"role": "user", "content": next_message})

        # call the API
        response = openai.ChatCompletion.create(
            model=self.model, messages=self.messages
        )

        # save the final message to ensure state is preserved
        self.messages.append(
            {
                "role": "assistant",
                "content": response["choices"][0]["message"]["content"],
            }
        )

        return response["choices"][0]["message"]["content"]

    # Generate a response to the given message, but stream the response as it is generated
    # adapted from https://github.com/trackzero/openai/blob/main/oai-text-gen-with-secrets-and-streaming.py
    # note: function is a generator, you must iterate over it to get the results
    def generate_response_stream(self, next_message):
        self.messages.append({"role": "user", "content": next_message})
        response = openai.ChatCompletion.create(
            model=self.model, messages=self.messages, stream=True
        )
        # event variables
        collected_chunks = []
        collected_messages = ""

        # capture event stream
        for chunk in response:
            collected_chunks.append(chunk)  # save the event response
            chunk_message = chunk["choices"][0]["delta"]  # extract the message
            if "content" in chunk_message: # make sure the message has "content" (the string we care about)
                message_text = chunk_message["content"]
                collected_messages += message_text
                yield message_text

        # once all chunks are received, save the final message
        self.messages.append({"role": "assistant", "content": collected_messages})
