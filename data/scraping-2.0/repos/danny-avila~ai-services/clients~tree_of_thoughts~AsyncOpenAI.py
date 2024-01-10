# clients\tree_of_thoughts\AsyncOpenAI.py
import os
import re
import openai
import logging
import asyncio
from tree_of_thoughts import AbstractLanguageModel

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class AsyncOpenAILanguageModel(AbstractLanguageModel):
    def __init__(self, api_key, strategy="cot", evaluation_strategy="value", api_base="", api_model="", enable_ReAct_prompting=True, stream_handler=None):
        self.session = openai.aiosession.get()
        self.stream_handler = stream_handler
        if api_key == "" or api_key == None:
            api_key = os.environ.get("OPENAI_API_KEY", "")
        if api_key != "":
            openai.api_key = api_key
        else:
            raise Exception("Please provide OpenAI API key")

        if api_base == "" or api_base == None:
            # if not set, use the default base path of "https://api.openai.com/v1"
            api_base = os.environ.get("OPENAI_API_BASE", "")
        if api_base != "":
            # e.g. https://api.openai.com/v1/ or your custom url
            openai.api_base = api_base
            print(f'Using custom api_base {api_base}')

        if api_model == "" or api_model == None:
            api_model = os.environ.get("OPENAI_API_MODEL", "")
        if api_model != "":
            self.api_model = api_model
        else:
            self.api_model = "text-davinci-003"
        print(f'Using api_model {self.api_model}')

        self.use_chat_api = 'gpt' in self.api_model

        # reference : https://www.promptingguide.ai/techniques/react
        self.ReAct_prompt = ''
        if enable_ReAct_prompting:
            self.ReAct_prompt = "Write down your observations in format 'Observation:xxxx', then write down your thoughts in format 'Thoughts:xxxx'."

        self.strategy = strategy
        self.evaluation_strategy = evaluation_strategy

    def stream_message(self, message):
        if self.stream_handler:
            self.stream_handler(message)

    async def openai_api_call_handler(self, prompt, max_tokens, temperature, k=1, stop=None):
        while True:
            try:
                if self.use_chat_api:
                    messages = [
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ]
                    response = await openai.ChatCompletion.acreate(
                        model=self.api_model,
                        messages=messages,
                        max_tokens=max_tokens,
                        temperature=temperature,
                    )
                else:
                    response = await openai.Completion.acreate(
                        engine=self.api_model,
                        prompt=prompt,
                        n=k,
                        max_tokens=max_tokens,
                        stop=stop,
                        temperature=temperature,
                    )
                # with open("openai.logs", 'a') as log_file:
                #     log_file.write("\n" + "-----------" +
                #                    '\n' + "Prompt : " + prompt+"\n")

                self.stream_message(f"\n-----------\nPrompt: {prompt}")
                return response
            except openai.error.RateLimitError as e:
                sleep_duration = os.environ.get("OPENAI_RATE_TIMEOUT", 30)
                print(
                    f'{str(e)}, sleep for {sleep_duration}s, set it by env OPENAI_RATE_TIMEOUT')
                await asyncio.sleep(sleep_duration)

    def openai_choice2text_handler(self, choice):
        if self.use_chat_api:
            text = choice['message']['content']
        else:
            text = choice.text.strip()
        return text

    async def generate_text(self, prompt, k):
        if self.use_chat_api:
            thoughts = []
            for _ in range(k):
                response = await self.openai_api_call_handler(prompt, 400, 0.5, k)
                text = self.openai_choice2text_handler(response.choices[0])
                thoughts += [text]
                self.stream_message(f"\n-----------\nThoughts: {thoughts}")
                # print(f'thoughts: {thoughts}')
            return thoughts

        else:
            response = await self.openai_api_call_handler(prompt, 300, 0.5, k)
            thoughts = [self.openai_choice2text_handler(
                choice) for choice in response.choices]
            return thoughts

    async def generate_thoughts(self, state, k, initial_prompt, rejected_solutions=None):
        if (type(state) == str):
            state_text = state
        else:
            state_text = '\n'.join(state)
        print("New state generating thought:", state, "\n\n")
        # prompt = f"Considering the thoughts you've had until now:\n\n{state_text}\n\nDevise the next coherent thought that will aid in advancing the reasoning process and achieving a solution to {initial_prompt}. Assess various scenarios, think unconventionally, anticipate potential challenges, and resolve any outstanding queries. Tap into your mind's full potential and make certain no open questions remain."
        prompt = f"""You are TreeofThoughts, a super-intelligent AI model devoted to helping Humans by any means necessary. Your purpose is to generate a series of solutions to comply with the user's instructions, you must generate solutions on the basis of determining the most reliable solution in the shortest amount of time, while taking rejected solutions into account and learning from them. 
        Considering the reasoning provided:\n\n
        ###'{state_text}'\n\n###
        Devise the best possible solution for the task: {initial_prompt}, Here are evaluated solutions that were rejected: 
        ###{rejected_solutions}###, 
        complete the {initial_prompt} without making the same mistakes you did with the evaluated rejected solutions. Be simple. Be direct. Provide intuitive solutions as soon as you think of them."""

        prompt += self.ReAct_prompt
        # print(prompt)
        thoughts = await self.generate_text(prompt, k)
        # print(thoughts)
        # print(f"Generated thoughts: {thoughts}")
        return thoughts

    async def generate_solution(self, initial_prompt, state, rejected_solutions=None):
        try:

            if isinstance(state, list):
                state_text = '\n'.join(state)
            else:
                state_text = state

            prompt = f"""You are TreeofThoughts, a super-intelligent AI model devoted to helping Humans by any means necessary. Your purpose is to generate a series of solutions to comply with the user's instructions, you must generate solutions on the basis of determining the most reliable solution in the shortest amount of time, while taking rejected solutions into account and learning from them. 
            Considering the reasoning provided:\n\n
            ###'{state_text}'\n\n###
            Devise the best possible solution for the task: {initial_prompt}, Here are evaluated solutions that were rejected: 
            ###{rejected_solutions}###, 
            complete the {initial_prompt} without making the same mistakes you did with the evaluated rejected solutions. Be simple. Be direct. Provide intuitive solutions as soon as you think of them."""
            answer = await self.generate_text(prompt, 1)
            print(f'Answer: {answer}')
            # print(thoughts)
            # print(f"General Solution : {answer}")
            return answer
        except Exception as e:
            logger.error(f"Error in generate_solutions: {e}")
            return None

    async def evaluate_states(self, states, initial_prompt):
        if not states:
            return {}

        if self.evaluation_strategy == 'value':
            state_values = {}
            for state in states:
                if (type(state) == str):
                    state_text = state
                else:
                    state_text = '\n'.join(state)
                print("We receive a state of type", type(
                    state), "For state: ", state, "\n\n")
                prompt = f""" To achieve the following goal: '{initial_prompt}', pessimistically value the context of the past solutions and more importantly the latest generated solution you had AS A FLOAT BETWEEN 0 AND 1\n
                    Past solutions:\n\n
                    {state_text}\n       
                    If the solutions is not directly concretely making fast progress in achieving the goal, give it a lower score.
                    Evaluate all solutions AS A FLOAT BETWEEN 0 and 1:\nDO NOT RETURN ANYTHING ELSE
                """

                response = await self.openai_api_call_handler(prompt, 50, 1)
                try:
                    value_text = self.openai_choice2text_handler(
                        response.choices[0])
                    # print(f'state: {value_text}')
                    self.stream_message(f"\n-----------\nValue Text: {value_text}")
                    # value = float(value_text)
                    matches = re.findall(r'[-+]?[0-9]*\.[0-9]+', value_text)
                    if matches:
                        value = float(matches[-1])  # Get the last match
                        print(f"Evaluated Thought Value: {value}")
                        self.stream_message(f"\n-----------\nEvaluated Thought Value: {value}")
                    else:
                        raise ValueError("No float found in value text")
                    print(f"Evaluated Thought Value: {value}")
                    self.stream_message(f"\n-----------\nEvaluated Thought Value: {value}")
                except ValueError:
                    self.stream_message(f"\n-----------\nValueError, defaulting to 0")
                    value = 0  # Assign a default value if the conversion fails
                state_values[state] = value
            return state_values

        elif self.evaluation_strategy == 'vote':
            states_text = '\n'.join([' '.join(state) for state in states])

            prompt = f"Given the following states of reasoning, vote for the best state utilizing an scalar value 1-10:\n{states_text}\n\nVote, on the probability of this state of reasoning achieveing {initial_prompt} and become very pessimistic very NOTHING ELSE"

            response = await self.openai_api_call_handler(prompt, 100, 1)

            print(f'state response: {response}')

            best_state_text = self.openai_choice2text_handler(
                response.choices[0])

            print(f"Best state text: {best_state_text}")

            best_state = tuple(best_state_text.split())

            print(f'best_state: {best_state}')

            return {state: 1 if state == best_state else 0 for state in states}

        else:
            raise ValueError(
                "Invalid evaluation strategy. Choose 'value' or 'vote'.")
