import openai
from paper.reAct import wikienv, wrappers
from urllib import request
from utils.diff_strings import print_clean_diff_strings

INSTRUCTION = """Solve a question answering task with interleaving Thought, Action, Observation steps. Thought can reason about the current situation, and Action can be two types: 
(1) Search[entity], which searches the exact entity on Wikipedia and returns the first paragraph if it exists. If not, it will return some similar entities to search.
(2) Finish[answer], which returns the answer and finishes the task.
Here are some examples.
"""

ONE_SHOT = """Question: Were Pavel Urysohn and Leonid Levin known for the same type of work?
Thought 1: I need to search Pavel Urysohn and Leonid Levin, find their types of work, then find if they are the same.
Action 1: Search[Pavel Urysohn]
Observation 1: Pavel Samuilovich Urysohn (February 3, 1898 â August 17, 1924) was a Soviet mathematician who is best known for his contributions in dimension theory.
Thought 2: Pavel Urysohn is a mathematician. I need to search Leonid Levin next and find its type of work.
Action 2: Search[Leonid Levin]
Observation 2: Leonid Anatolievich Levin is a Soviet-American mathematician and computer scientist. 
Thought 3: Leonid Levin is a mathematician and computer scientist. So Pavel Urysohn and Leonid Levin have the same type of work. 
Action 3: Finish[yes]"""

ALT_SHOT = """Question: What profession does Nicholas Ray and Elia Kazan have in common?
Thought 1: I need to search Nicholas Ray and Elia Kazan, find their professions, then find the profession they have in common.
Action 1: Search[Nicholas Ray]
Observation 1: Nicholas Ray (born Raymond Nicholas Kienzle Jr., August 7, 1911 – June 16, 1979) was an American film director, screenwriter, and actor best known for the 1955 film Rebel Without a Cause.
Thought 2: Professions of Nicholas Ray are director, screenwriter, and actor. I need to search Elia Kazan next and find his professions.
Action 2: Search[Elia Kazan]
Observation 2: Elia Kazan was an American film and theatre director, producer, screenwriter and actor.
Thought 3: Professions of Elia Kazan are director, producer, screenwriter, and actor. So profession Nicholas Ray and Elia Kazan have in common is director, screenwriter, and actor.
Action 3: Finish[director, screenwriter, actor]
"""

class ReActWiki(object):
    def __init__(self, **kwargs) -> None:
        self.print_prompt = kwargs.get('print_prompt', False)
        self.prompt = INSTRUCTION + ONE_SHOT + '\n'
        env = wikienv.WikiEnv()
        env = wrappers.HotPotQAWrapper(env, split="dev")
        self.env = wrappers.LoggingWrapper(env)
        self.preprinted_prompt = ''

    def llm(self, prompt, stop=["\n"]):
        response = openai.Completion.create(
            model="text-davinci-003",
            prompt=prompt,
            temperature=0,
            max_tokens=100,
            top_p=1,
            frequency_penalty=0.0,
            presence_penalty=0.0,
            stop=stop
        )
        return response["choices"][0]["text"]

    def step(self, action):
        attempts = 0
        while attempts < 10:
            try:
                return self.env.step(action)
            except request.exceptions.Timeout:
                attempts += 1

    def __call__(self, question):
        prompt = self.prompt
        prompt += question + '\n'
        n_calls, n_badcalls = 0, 0

        for i in range(1, 8):
            printed_prompt = prompt
            n_calls += 1
            thought_action = self.llm(
                prompt + f"Thought {i}:", stop=[f"\nObservation {i}:"])
            try:
                thought, action = thought_action.strip().split(
                    f"\nAction {i}: ")
            except:
                print('ohh...', thought_action)
                n_badcalls += 1
                n_calls += 1
                thought = thought_action.strip().split('\n')[0]
                action = self.llm(
                    prompt + f"Thought {i}: {thought}\nAction {i}:", stop=[f"\n"]).strip()
            obs, r, done, info = self.step(action[0].lower() + action[1:])
            obs = obs.replace('\\n', '')
            step_str = f"Thought {i}: {thought}\nAction {i}: {action}\nObservation {i}: {obs}\n"
            prompt += step_str

            self.preprinted_prompt = printed_prompt
            printed_prompt += step_str

            if self.print_prompt:
                print_clean_diff_strings(self.preprinted_prompt, printed_prompt)
                print('\n')
                
            if done:
                break
        if not done:
            obs, r, done, info = self.step("finish[]")
        info.update({'n_calls': n_calls, 'n_badcalls': n_badcalls, 'traj': prompt})
        return r, info

    def run(self, question):
        r, info = self(question)
        return info['answer']