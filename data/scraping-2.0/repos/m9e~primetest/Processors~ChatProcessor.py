import openai
import logging
import json
import os
import tiktoken

# Set your API key as an environment variable
openai.api_key = os.environ["OPENAI_API_KEY"]

logging.basicConfig(level=logging.DEBUG)

class ChatProcessor:

    # token count where we begin to distill interactions
    TOKEN_THRESHOLD = 4096

    # token count under which we don't distill a given message
    MSG_THRESHOLD = 4096
    MAX_TOKENS = 4096

    # do we stream responses?
    STREAM = True
    CREATE_TITLES = False

    def __init__(self, model="gpt-3.5-turbo", temperature=0.0):
        self.model = model
        self.title = "New Chat"
        self.streaming = False
        self.temperature = temperature

        # now that we can do gpt-4, ensure we set max tokens correctly
        if model.startswith('gpt-4'):
            self.MAX_TOKENS = 8192
        if model.startswith('gpt-4-32k'):
            self.MAX_TOKENS = 32768

        # does not affect manual calls to reduce()
        self.reduction_enabled = False

        self.multi = False
        self.reducer_messages = None
        self.reducer_prompt = "The text below is a message I received from ChatGPT in another chat. I want to use it in the chat history. I want to be able to send this text back to ChatGPT, but I want to minimize the number of tokens. Please distill this message into the smallest possible form that ChatGPT would interpret in a syntactically and semantically identical way. It will not be human read, and so any format that minimizes the number of tokens is acceptable. If it lowers the number of tokens, feel free to remove punctuation, new lines, articles of speech and anything else that does not impact the ability of ChatGPT to interpret the distilled version identically. The message: "
        self.title_prompt = "Based on the messages before now, please provide a 4 word or less title for this conversation, appropriate for a nav bar. answer with only the title and no other text, formatting, or explanation"
        self.start_messages = [
            {"role": "system", "content": "You are a helpful assistant. Answer as concisely as possible."},
        ]
        self.messages = self.start_messages

    def strtokens(self, text):
        encoding = tiktoken.encoding_for_model(self.model)
        return 4 + len(encoding.encode(text))

    def tokens(self, messages):
        token_count = 1
        for m in messages:
            token_count += 4
            for k, v in m.items():
                token_count += self.strtokens(v)
                if k == "name":
                    token_count += -1
        token_count += 2
        return token_count

    def update_title(self):
        if not self.CREATE_TITLES or self.title != "New Chat":
            return

        logging.debug("Updating title")
        try:
            response = openai.ChatCompletion.create(
                temperature=0.0,
                model='gpt-3.5-turbo',
                max_tokens=1000,
                # messages as a copy of self.messages plus the title prompt
                messages=self.messages + [{"role": "user", "content": self.title_prompt}]
            )
            logging.debug(str(response))
            self.title = response['choices'][0]['message']['content']
            logging.debug("Title updated to %s" % self.title)
        except Exception as e:
            # verbosely dump trace for debugging
            logging.debug(e, exc_info=True)
            self.title = "No Title available"
            
        return


    def reduce(self):
        token_count = self.tokens(self.messages)
        if token_count > self.TOKEN_THRESHOLD:
            print(
                f"token count of {token_count} exceeds {self.TOKEN_THRESHOLD}, must reduce")
            # Save the introduction
            new_messages = [self.start_messages[0]]

            # Message distiller block
            for m in self.messages[1:]:
                mtokens = 0
                for k, v in m.items():
                    mtokens += self.strtokens(v)
                    if k == "name":
                        mtokens -= 1

                if mtokens > self.MSG_THRESHOLD:
                    self.reducer_messages = [
                        {"role": "system", "content": "You distill text to optimize token counts. Avoid losing meaningful context while distilling."},
                    ]
                    message_str = "\n\n".join(
                        [f"{k}: {v}" for k, v in m.items()])
                    self.reducer_messages.append(
                        {"role": "user", "content": self.reducer_prompt + "\n\n" + message_str})

                    tc = self.tokens(self.reducer_messages)
                    response = openai.ChatCompletion.create(
                        temperature=self.temperature,
                        # always use gpt-3.5-turbo to reduce, since primary use case is to reduce gpt-4 tokens for 3.5
                        model='gpt-3.5-turbo',
                        messages=self.reducer_messages,
                        max_tokens=(self.MAX_TOKENS - tc)
                    )
                    self.last_response = response['choices'][0]['message']['content']
                    new_messages.append(
                        {"role": m["role"], "content": self.last_response})
                else:
                    new_messages.append(m)

            print("Distilled\n%s\n\nto\n\n%s" %
                  (str(self.messages), str(new_messages)))

            print("Reduced %d tokens to %d" %
                  (token_count, self.tokens(new_messages)))

            self.messages = new_messages

    def generate_response(self, prompt, stream_callback=None):
        self.messages.append({"role": "user", "content": prompt})
        self.last_response = None
        self.last_response_reason = None

        if self.reduction_enabled:
            self.reduce()

        token_count = self.tokens(self.messages)

        if self.STREAM:
            self.last_response = ""
            response = openai.ChatCompletion.create(
                temperature=self.temperature,
                model=self.model,
                messages=self.messages,
                max_tokens=(self.MAX_TOKENS - token_count),
                stream=True
            )
            for chunk in response:
                # ignore chunks without content, which will be roles,
                # which for now can only be assistant in theory anyhow
                if chunk['choices'][0]['delta'].get('content'):
                    self.last_response = self.last_response + chunk['choices'][0]['delta']['content']
                    if stream_callback:
                        stream_callback(chunk)
                    if chunk['choices'][0].get('finish_reason'):
                        self.last_response_reason = chunk['choices'][0]['finish_reason']

            self.messages.append(
                {"role": "assistant", "content": self.last_response})
        else:
            response = openai.ChatCompletion.create(
                temperature=self.temperature,
                model=self.model,
                messages=self.messages,
                max_tokens=(self.MAX_TOKENS - token_count)
            )
            self.last_response = response['choices'][0]['message']['content']
            self.last_response_reason = response['choices'][0]['finish_reason']
            self.messages.append(
                {"role": "assistant", "content": self.last_response})

        self.update_title()
        return {'last_response': self.last_response, 'chat_title': self.title}

    def dump_state(self, file=None):
        print("Dumping state to %s" % (file if file else 'state.json'))
        with open(file if file else 'state.json', 'w') as f:
            # dump self.messages and self.title
            f.write(json.dumps({"messages": self.messages, "title": self.title}))

    def restore_state(self, file=None):
        try:
            with open(file if file else 'state.json', 'r', encoding='utf-8') as f:
                data = f.read()
                state = json.loads(data)
                self.messages = state["messages"]
                self.title = state["title"]
        except Exception as e_error:
            print(f"Exception occurred restoring state: {str(e_error)}")
