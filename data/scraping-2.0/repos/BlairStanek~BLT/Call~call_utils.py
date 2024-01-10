import os, time, re

# This class abstracts away the details of which LLM is being called
class LLMChat:
    def __init__(self, model:str, system_prompt=None):
        self.model = model
        self.history = []
        self.system_prompt = system_prompt
        self.bison_chat_model = None
        self.bison_chat = None
        self.legacy_davinci_tokenizer = None
        self.claude_model = None

    # Call this to actually pass chat text; it returns the response
    def chat(self, prompt):
        response_text = None
        if "gpt-3" in self.model or "gpt-4" in self.model or "text-davinci-003" == self.model:
            # we import lazily (ie, not at top) so those using only 1 of the APIs
            # don't have to install the python support for the others
            import  openai
            client = openai.OpenAI(api_key=os.getenv("GPT_API_KEY"))

            if len(self.history) == 0 and self.system_prompt is not None:
                self.history.append({"role": "system", "content": self.system_prompt})
            self.history.append({"role": "user", "content": prompt})
            worked = False
            while not worked:
                try:
                    if "text-davinci-003" not in self.model:  # We use this code if NOT text-davinci-003, which is most of the time
                        response = client.chat.completions.create(
                            model=self.model,
                            messages=self.history,
                            temperature=0.0,
                            timeout=30
                        )
                        # temperature = 0.0,
                        # request_timeout = 30

                        response_text = response.choices[0].message.content
                        worked = True
                    else: # if text-davinci-003
                        # we import lazily (ie, not at top) so those using only 1 of the APIs
                        # don't have to install the python support for the others
                        import tiktoken

                        prompt = ""
                        for message in self.history:  # basically just concat everything
                            if message["role"] != "system":
                                prompt += message["content"]

                        # We need to calculate the prompt remaining, since otherwise it's just 16 tokens!
                        if self.legacy_davinci_tokenizer == None:
                            self.legacy_davinci_tokenizer = \
                                tiktoken.encoding_for_model("text-davinci-003")  # may be used for max_tokens
                        gpt3_encoding = self.legacy_davinci_tokenizer.encode(prompt)
                        BUFFER = 10
                        MAX_GPT3_TOKENS = 4000
                        max_tokens = MAX_GPT3_TOKENS - len(gpt3_encoding) - BUFFER

                        response = client.completions.create(
                            model="text-davinci-003",
                            prompt=prompt,
                            temperature=0.0,
                            max_tokens=max_tokens
                        )
                        response_text = response.choices[0].text
                        worked = True
                    self.history.append({"role": "assistant", "content": response_text})

                except openai.InternalServerError:
                    print("InternalServerError error, retrying in 5s.")
                    time.sleep(5)
                except openai.RateLimitError:
                    print("RateLimitError error, retrying in 5s.")
                    time.sleep(5)
                except openai.APIConnectionError:
                    print("APIConnectionError error, retrying in 5s.")
                    time.sleep(5)
                except openai.APIError:
                    print("APIError error, retrying in 5s.")
                    time.sleep(5)
                # except openai.Timeout:
                #     print("Timeout error, retrying in 5s.")
                #     time.sleep(5)
            assert worked
        elif "chat-bison@001" == self.model: # This is a Google model
            # we import lazily (ie, not at top) so those using only 1 of the APIs
            # don't have to install the python support for the others
            from vertexai.language_models import ChatModel
            from google.api_core.exceptions import ResourceExhausted
            # import google.generativeai as palm # We now use the APIs above

            # You must have already configured a Google Cloud project and set up authorization with billing
            # to call bison in this way.
            worked = False
            while not worked:
                try:
                    if self.bison_chat_model is None:
                        self.bison_chat_model = ChatModel.from_pretrained(self.model)
                        self.bison_chat = self.bison_chat_model.start_chat(temperature=0.0)
                    self.bison_chat.send_message(prompt)
                    response_text = self.bison_chat.message_history[-1].content
                    worked = True
                except ResourceExhausted:
                    print("Resource Exhausted, retrying in 5s.")
                    time.sleep(5)
            assert worked

        elif "chat-bison-32k" == self.model: # This is a Google model
            # we import lazily (ie, not at top) so those using only 1 of the APIs
            # don't have to install the python support for the others
            from vertexai.preview.language_models import ChatModel
            from google.api_core.exceptions import ResourceExhausted
            # import google.generativeai as palm # We now use the APIs above

            # You must have already configured a Google Cloud project and set up authorization with billing
            # to call bison in this way.
            worked = False
            while not worked:
                try:
                    if self.bison_chat_model is None:
                        self.bison_chat_model = ChatModel.from_pretrained(self.model)
                        self.bison_chat = self.bison_chat_model.start_chat(temperature=0.0)
                    self.bison_chat.send_message(prompt)
                    response_text = self.bison_chat.message_history[-1].content
                    worked = True
                except ResourceExhausted:
                    print("Resource Exhausted, retrying in 5s.")
                    time.sleep(5)
            assert worked

        elif "claude" in self.model: # This is a Google model
            # we import lazily (ie, not at top) so those using only 1 of the APIs
            # don't have to install the python support for the others
            from anthropic import Anthropic, HUMAN_PROMPT, AI_PROMPT
            self.history.append({"role": "user", "content": prompt})

            worked = False
            while not worked:
                try:
                    if self.claude_model is None:
                        self.claude_model = Anthropic() # you must have ANTHROPIC_API_KEY already in your environment

                    claude_prompt = ""
                    for message in self.history:  # basically just concat everything
                        if message["role"] == "user":
                            claude_prompt += HUMAN_PROMPT
                            claude_prompt += message["content"]
                        elif message["role"] == "assistant":
                            claude_prompt += AI_PROMPT
                            claude_prompt += message["content"]
                    claude_prompt += AI_PROMPT

                    completion = self.claude_model.completions.create(
                        model=self.model,
                        max_tokens_to_sample=3000,
                        prompt=claude_prompt,
                        temperature=0.0
                    )
                    response_text = completion.completion
                    self.history.append({"role": "assistant", "content": response_text}) # store for future

                    worked = True
                except:
                    print("Error, retrying in 5s.")
                    time.sleep(5)
            assert worked
        else:
            assert False, "model not supported:" + self.model


        return response_text


# This fixes some common problems that cause incorrect negative results
def standardize_text(text_in:str) -> str:
    return text_in.replace("‒", "-").replace("–", "-").replace("—", "-")

# These are all used in error analyses
NOT_PARALLEL = "not parallel"
NOT_FOUND = "not found"
MULTIPLE_SOMEWRONG = "multiple answers"
WRONG_LINE_MULTIPLE = "multiple wrong lines"
WRONG_LINE_SUBSET = "subset wrong line"
INCORRECT_FORMAT = "incorrect format"
WRONG_LINE = "wrong line" # used for transcripts
SUBSET = "subset"
SUPERSET = "superset"
CHILD_OF_CORRECT = "child of correct"
CHILD_OF_WRONG = "child of wrong" # the term returned is a child of a wrong term
SECTIONNUM_OMITTED = "section number omitted"
OMITTED_DUMMY_AMENDMENT = "omitted dummy amendment"

WRONG_ERRORNAME_PREFIX = "wrong " # appended to the subdivision types below
subdivision_types = ["subsection", "paragraph", "subparagraph",
                     "clause", "subclause", "item", "subitem", "subsubitem"]
WRONG_SECTION = WRONG_ERRORNAME_PREFIX + "section"


# errors can be described in either absolute terms (e.g. "wrong subpargraph") or in
# relative terms (e.g. "wrong item two levels above target").
def analyze_error(correct_cite:str, incorrect_cite:str):
    # Build the regex
    match_re = ""
    for subd in subdivision_types:
        match_re += "(?P<" + subd + ">\(\w+\))?" # create a group for each
    match_re += "$" # forces end of line

    rv = []
    cor = re.match(match_re, correct_cite)
    incor = re.match(match_re, incorrect_cite)
    if cor is None or incor is None:
        return [INCORRECT_FORMAT]

    # Check to see if there's a simple failure of parallelism
    for group in subdivision_types:
        if (cor.group(group) is None) != (incor.group(group) is None):
            return [NOT_PARALLEL]
        if cor.group(group) != incor.group(group):
            rv.append(WRONG_ERRORNAME_PREFIX + group)
    return rv

# When there are multiple possible errors, we need to rank the most plausible
# to record what the error was
def seconderrors_more_plausible(errors1:list, errors2:list) -> bool:
    if errors1 == [CHILD_OF_WRONG] and errors2 != [CHILD_OF_WRONG]:
        return True
    if errors1 == [INCORRECT_FORMAT] and errors2 != [INCORRECT_FORMAT]:
        return True
    if errors1 == [NOT_PARALLEL] and errors2 != [NOT_PARALLEL]:
        return True
    if errors1 == [WRONG_SECTION] and errors2 != [WRONG_SECTION]:
        return True
    if len(errors1) > len(errors2):
        return True
    return False

def strip_line_to_text(intext:str) -> str:
    rv = intext.strip()
    if rv.startswith("("):
        return rv[rv.find(")")+1:].lstrip()
    if rv.startswith("["):
        return rv[rv.find("]")+1:].lstrip()
    if re.match("[1-9][0-9]?:", rv):
        return rv[rv.find(":")+1:].lstrip()
    return rv

# if __name__ == "__main__":