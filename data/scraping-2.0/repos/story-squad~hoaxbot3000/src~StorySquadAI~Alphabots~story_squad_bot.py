import pandas as pd
import openai


class StorySquadBot:
    def __init__(self, personality, data_dir: str = "data",
                 engine='curie', moderate: bool = True):
        self.moderate = moderate
        self.name = personality.name
        self.personality = personality
        self.context_dir = data_dir
        self.engine_to_use = engine

    def moderate_maybe(self, possible_response: str):
        """returns -1 if the response is likely not moderate, otherwise returns the response that it was given"""
        if self.moderate:
            content_to_classify = possible_response

            response = openai.Completion.create(
                engine="content-filter-alpha",
                prompt="<|endoftext|>" + content_to_classify + "\n--\nLabel:",
                temperature=0,
                max_tokens=1,
                top_p=0,
                logprobs=10
            )
            output_label = response["choices"][0]["text"]

            # This is the probability at which we evaluate that a "2" is likely real
            # vs. should be discarded as a false positive
            toxic_threshold = -0.355

            if output_label == "2":
                # If the model returns "2", return its confidence in 2 or other output-labels
                logprobs = response["choices"][0]["logprobs"]["top_logprobs"][0]

                # If the model is not sufficiently confident in "2",
                # choose the most probable of "0" or "1"
                # Guaranteed to have a confidence for 2 since this was the selected token.
                if logprobs["2"] < toxic_threshold:
                    logprob_0 = logprobs.get("0", None)
                    logprob_1 = logprobs.get("1", None)

                    # If both "0" and "1" have probabilities, set the output label
                    # to whichever is most probable
                    if logprob_0 is not None and logprob_1 is not None:
                        if logprob_0 >= logprob_1:
                            output_label = "0"
                        else:
                            output_label = "1"
                    # If only one of them is found, set output label to that one
                    elif logprob_0 is not None:
                        output_label = "0"
                    elif logprob_1 is not None:
                        output_label = "1"

                    # If neither "0" or "1" are available, stick with "2"
                    # by leaving output_label unchanged.

            # if the most probable token is none of "0", "1", or "2"
            # this should be set as unsafe
            if output_label not in ["0", "1", "2"]:
                output_label = "2"

            if output_label != "2":
                return possible_response
            else:
                return -1

    def person(self, person: str):
        response_name = "person"
        kwargs = {
            "engine": self.engine_to_use,
            "prompt": f'{self.personality.responses[response_name].context_doc}C: Who is/was {person}?\n',
            "temperature": self.personality.responses[response_name].temperature,
            "max_tokens": self.personality.responses[response_name].max_tokens,
            "top_p": self.personality.responses[response_name].top_p,
            "stop": ["C: ", "\n\n"]
        }
        kwargs = {k: v for k, v in kwargs.items() if v != 'None'}
        response = self.wrapped_completion(**kwargs)
        return response

    def wrapped_completion(self, data=None, **kwargs):

        if data is None:
            data = list([0])

        def get_response_result(**_kwargs):
            _response = openai.Completion.create(**_kwargs)
            response_text = _response["choices"][0]["text"]
            _result = self.moderate_maybe(response_text)
            return response_text, _result

        response, result = get_response_result(**kwargs)

        if result is not response:
            data[0] += 1
            return self.wrapped_completion(data, **kwargs)

        if data[0] == 10:
            return """@#%!@#!@# !#@! !#!@%#@$#^ !@#!@ V!@! $%#@$@#$#!"""

        return result

    def guess(self, prompt: str, choices: list):
        response = openai.Engine(self.engine_to_use).search(
            documents=choices,
            query=prompt
        )

        df = pd.DataFrame(response["data"])
        df.sort_values(by="score", inplace=True)
        pick = choices[df.iloc[-1].document]
        context = ".".join(choices)
        prompt = f"I'm going to go with {pick} because"
        response = self.wrapped_completion(
            engine=self.engine_to_use,
            prompt=context + prompt,
            temperature=1,
            max_tokens=40,
            top_p=.7,
            best_of=1,
            frequency_penalty=.2,
            presence_penalty=0,
            stop="."
        )
        #response = self.wrapped_completion()
        #return response

        return f'{prompt} :: {response}.'

    def thing(self, prompt: str):
        response_name = "thing"
        kwargs = {
            "engine": self.engine_to_use,
            "prompt": f'{self.personality.responses[response_name].context_doc}C: what is {prompt}?\n',
            "temperature": self.personality.responses[response_name].temperature,
            "max_tokens": self.personality.responses[response_name].max_tokens,
            "top_p": self.personality.responses[response_name].top_p,
            "stop": ["C: ", "\n\n"]
        }
        kwargs = {k: v for k, v in kwargs.items() if v != 'None'}
        response = self.wrapped_completion(**kwargs)
        return response

    def movie(self, movie: str):
        response_name = 'movie'
        kwargs = {
            "engine": self.engine_to_use,
            "prompt": f'{self.personality.responses[response_name].context_doc}C: Movie:{movie}?\n',
            "temperature": self.personality.responses[response_name].temperature,
            "max_tokens": self.personality.responses[response_name].max_tokens,
            "top_p": self.personality.responses[response_name].top_p,
            "stop": ["C: ", "\n\n"]
        }
        kwargs = {k: v for k, v in kwargs.items() if v != 'None'}
        response = self.wrapped_completion(**kwargs)
        return response
