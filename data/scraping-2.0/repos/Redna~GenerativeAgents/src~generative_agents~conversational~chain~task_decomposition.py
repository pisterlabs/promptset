import re
from pydantic import BaseModel
from generative_agents import global_state
from generative_agents.conversational.llm import llm
from langchain import LLMChain, PromptTemplate

# TODO rewrite few shot prompt to match the new identity set
_template = """
Describe subtasks in 5 min increments.
---
Name: Kelly Bronson
Age: 35
Backstory: Kelly always wanted to be a teacher, and now she teaches kindergarten. During the week, she dedicates herself to her students, but on the weekends, she likes to try out new restaurants and hang out with friends. She is very warm and friendly, and loves caring for others.
Personality: sweet, gentle, meticulous
Location: Kelly is in an older condo that has the following areas: [kitchen, bedroom, dining, porch, office, bathroom, living room, hallway].
Currently: Kelly is a teacher during the school year. She teaches at the school but works on lesson plans at home. She is currently living alone in a single bedroom condo.
Daily plan requirement: Kelly is planning to teach during the morning and work from home in the afternoon.s

Today is Saturday May 10. From 08:00am ~09:00am, Kelly is planning on having breakfast, from 09:00am ~ 12:00pm, Kelly is planning on working on the next day's kindergarten lesson plan, and from 12:00 ~ 13pm, Kelly is planning on taking a break. 
In minimum 5 minutes increments, list the subtasks Kelly does when Kelly is working on the next day's kindergarten lesson plan from 09:00am ~ 12:00pm (total duration in minutes: 180):
1) Kelly is reviewing the kindergarten curriculum standards. (duration in minutes: 15, minutes left: 165)
2) Kelly is brainstorming ideas for the lesson. (duration in minutes: 30, minutes left: 135)
3) Kelly is creating the lesson plan. (duration in minutes: 30, minutes left: 105)
4) Kelly is creating materials for the lesson. (duration in minutes: 30, minutes left: 75)
5) Kelly is taking a break. (duration in minutes: 15, minutes left: 60)
6) Kelly is reviewing the lesson plan. (duration in minutes: 30, minutes left: 30)
7) Kelly is making final changes to the lesson plan. (duration in minutes: 15, minutes left: 15)
8) Kelly is printing the lesson plan. (duration in minutes: 10, minutes left: 5)
9) Kelly is putting the lesson plan in her bag. (duration in minutes: 5, minutes left: 0)
---
{identity}

Today is {today}. {task_context}
In minimum 5 minutes increments, list the subtasks {name} does when {name} is {task_description} from {task_start_time} ~ {task_end_time} (total duration in minutes: {task_duration}):
1) {name} is"""


class TaskDecomposition(BaseModel):

    name: str
    identity: str
    today: str
    task_context: str
    task_description: str
    task_start_time: str
    task_end_time: str
    task_duration: str

    async def run(self):
        _prompt = PromptTemplate(input_variables=["name",
                                                  "identity",
                                                  "today",
                                                  "task_context",
                                                  "task_description",
                                                  "task_start_time",
                                                  "task_end_time",
                                                  "task_duration"],
                                 template=_template)

        _task_decomposition_chain = LLMChain(prompt=_prompt, llm=llm, llm_kwargs={
            "max_new_tokens": 550,
            "do_sample": True,
            "top_p": 0.95,
            "top_k": 60,
            "temperature": 0.4,
            "cache_key": f"2task_decomposition_{self.name}_{global_state.tick}"}, verbose=True)

        completion = await _task_decomposition_chain.arun(name=self.name,
                                                          identity=self.identity,
                                                          today=self.today,
                                                          task_context=self.task_context,
                                                          task_description=self.task_description,
                                                          task_start_time=self.task_start_time,
                                                          task_end_time=self.task_end_time,
                                                          task_duration=self.task_duration)

        pattern = rf"\d+\) {self.name} is (.*) \(duration in minutes: (\d+).*"
        return [list(task) for task in re.findall(pattern, completion)]
