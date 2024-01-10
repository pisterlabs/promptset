#!/user/bin/env python3

import openai

import re
import time
from itertools import repeat
from multiprocessing import Pool

import app_config

_MAX_SAMPLE_SIZE = 100

# For parallelizing value-specific summary queries.  Check GPT-3 API rate limit.
_NUM_PROCESSES = 10


class OfflineModel:
    def prompt_examples(self, df, column, facet_column, facet_val):
        df = df[df[column].str.len() > 0]
        if facet_column:
            df = df[
                df[facet_column].str.contains("(^|;)" + re.escape(facet_val) + "($|;)")
            ]
        return df[column]

    def canned_answer(self, examples):
        if len(examples) == 0:
            return "None of the respondees answered this question."
        elif len(examples) == 1:
            return 'There was just one nonempty answer: "%s"' % (list(examples)[0])
        else:
            return "No GPT-3 model available to summarize the %d answers" % (
                len(examples)
            )

    def get_summary(self, df, column, facet_column=None, facet_val=None, prompt=None, temperature=0.0):
        examples = self.prompt_examples(df, column, facet_column, facet_val)
        return {
            "instructions": "No GPT-3 model available",
            "answer": self.canned_answer(examples),
            "nonempty_responses": len(examples),
            "sample_size": len(examples),
            "facet_column": facet_column,
            "facet_val": facet_val,
        }

    def get_summaries(
        self, df, question_column, facet_column, facet_values, prompt=None, temperature=0.0
    ):
        return [
            self.get_summary(df, question_column, facet_column, x) for x in facet_values
        ]


class LiveGptModel(OfflineModel):
    def get_summary(
        self,
        df,
        column,
        facet_column=None,
        facet_val=None,
        prompt=app_config.DEFAULT_PROMPT,
        temperature=0.0,
    ):
        model = app_config.PROMPTS[prompt]["model"]
        if column == app_config.COLUMN_NAME_FOR_TEXT_FILES:
            # For single-column text files, do not label the examples
            preamble = ""
        else:
            preamble = 'Here are some responses to "%s":\n' % (column)
        instructions = app_config.PROMPTS[prompt]["prompt"]
        nonempty_responses = self.prompt_examples(df, column, facet_column, facet_val)

        # See https://help.openai.com/en/articles/4936856-what-are-tokens-and-how-to-count-them
        max_words = (
            app_config.MAX_TOKENS[model] / 1.5 - len(preamble) - len(instructions) - 500
        )

        examples = None

        max_sample_size = _MAX_SAMPLE_SIZE
        while examples is None or len("\n".join(examples).split()) > max_words:
            examples = nonempty_responses.sample(
                min(max_sample_size, len(nonempty_responses)),
                random_state=(int(temperature*1000))
            )
            # Keep reducing number of examples until it fits in max_words
            if max_sample_size > 10:
                max_sample_size -= 10
            else:
                max_sample_size -= 1

        if len(examples) <= 1:
            answer = self.canned_answer(examples)
        else:
            prompt_str = (
                preamble
                + "\n".join([("- " + s) for s in examples])
                + "\n\n"
                + instructions
                + "\n"
            )
            response = run_completion_query(prompt_str, model=model, temperature=temperature)
            answer = set([c["text"] for c in response["choices"]])
            answer = "\n".join(list(answer))
        return {
            "instructions": instructions,
            "answer": answer,
            "nonempty_responses": len(nonempty_responses),
            "sample_size": len(examples),
            "facet_column": facet_column,
            "facet_val": facet_val,
        }

    def get_summaries(
        self,
        df,
        question_column,
        facet_column,
        facet_values,
        prompt=app_config.DEFAULT_PROMPT,
        temperature=0.0,
    ):
        """Get a summary for each value of a facet."""
        if True:
            # Serial
            return [
                self.get_summary(df, question_column, facet_column, x, prompt=prompt)
                for x in facet_values
            ]
        else:
            # Parallel
            with Pool(_NUM_PROCESSES) as pool:
                return list(
                    pool.starmap(
                        self.get_summary,
                        zip(
                            repeat(df),
                            repeat(question_column),
                            repeat(facet_column),
                            facet_values,
                            repeat(prompt),
                        ),
                    )
                )


def get_config():
    return {"llm": app_config.USE_GPT3 and LiveGptModel() or OfflineModel()}


def run_completion_query(prompt, model="text-davinci-003", num_to_generate=1, temperature=0.0):
    tries = 0
    while tries < 3:
        try:
            if model.startswith("gpt"):
                # It's one of the chat-tuned models
                response = openai.ChatCompletion.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=temperature,
                    max_tokens=500,
                )
                # Hack: Move the response text to the old API's expected location
                response["choices"][0]["text"] = response["choices"][0]["message"]["content"]
            else:
                response = openai.Completion.create(
                    model=model,
                    prompt=prompt,
                    temperature=temperature,
                    n=num_to_generate,
                    stop=["\n\n"],
                    max_tokens=300,
                )
            return response
        except (openai.error.RateLimitError, openai.error.APIError) as e:
            print(e)
            tries += 1
            time.sleep(10)
