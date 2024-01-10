# pylint: disable=missing-module-docstring
# pylint: disable=missing-class-docstring
# pylint: disable=missing-function-docstring
# pylint: disable=multiple-statements

import abc
import sys
import json
import string       as s
import typing       as t
import time         as tm
import dataclasses  as dc
import subprocess   as sp
import langchain    as lc


# actors creating the magic
ai: "AI"
bot: "Bot"
user: "User"


# =============================================================================
# Skills database and management


# all the skills of the bot
skills: list["Skill"] = []


# these actions are used to process a task
actionsAfterTask   = ["TaskList", "UserQuestions", "SearchQuery", "Options"]

# these actions are used to process a result
actionsAfterResult = ["Summary", "Shell", "UserQuestions", "SearchQuery", "Options"]


# examples of actions
T = t.TypeVar("T", bound = "Action")
OneOrList = t.Union[T, list[T]]

# type aliases for skills input and output
SkillInput  = t.TypeVar("SkillInput",  bound = "Action")
SkillOutput = t.TypeVar("SkillOutput", bound = "Action")


class Skill(t.Generic[SkillInput, SkillOutput], abc.ABC):
    """Base class for skills of the bot."""

    @property
    def name(self) -> str:
        """Returns the name of the skill."""
        return self.__class__.__name__

    @property
    def inputActionType(self) -> str:
        """Returns the type of the input action."""
        return self.__orig_bases__[0].__args__[0].__name__ # type: ignore

    @property
    def outputActionType(self) -> str:
        """Returns the type of the output action."""
        return self.__orig_bases__[0].__args__[1].__name__ # type: ignore

    # registers the skill in the database
    def __init_subclass__(cls, *args, **kwargs) -> None:
        """Registers the skill in the database."""
        super().__init_subclass__(*args, **kwargs)
        skills.append(cls())

    # override this method to provide 
    @abc.abstractmethod
    def UseWhen(self) -> str:
        """Returns a hint to explain when to use this skill."""
        raise NotImplementedError()

    # override this method to provide an example of the input
    @abc.abstractmethod
    def InputExample(self) -> OneOrList[SkillInput]:
        """Returns a JSON example of the input action."""
        raise NotImplementedError()

    # override this method to implement the skill
    @abc.abstractmethod
    def Use(self, input_: SkillInput) -> SkillOutput:
        """Uses the skill with the given input."""
        raise NotImplementedError()

    # override this method to provide a follow-up message
    @abc.abstractmethod
    def FollowUpMessage(self, output: SkillOutput) -> str:
        """Returns a follow-up message for the AI."""
        raise NotImplementedError()

    # override this method to specify follow-up actions
    @abc.abstractmethod
    def FollowUpActions(self) -> list[str]:
        """Returns a list of follow-up actions for the AI."""
        raise NotImplementedError()


    def FormattedInputExamples(self) -> str:
        """Gets the formatted input examples for the AI."""
        example = self.InputExample()

        # if the input is a list, we show each example separately
        if isinstance(example, list):
            return "\n".join([
                f"Example {index + 1}:" + json.dumps(ex)
                for index, ex in enumerate(example)
            ])

        # otherwise we show the example as a whole
        return "Example: " + json.dumps(example)


class Action(t.TypedDict, total = True):
    """Base class for things to be done by the AI."""
    code: str           # unique code for easy identification
    actionType: str     # type of action, to select the right function


def GetSkill(input_: Action) -> "Skill":
    """Gets the skill to use with the given input."""

    # gets the type of action and the skill to use
    actionType = input_["actionType"]
    skill = next((s for s in skills if s.inputActionType == actionType), None)

    # raises an error if the skill is not found
    if skill is None:
        raise ValueError(f"Skill not found with input action type: {actionType}")

    return skill


# =============================================================================
# Managing tasks and actions


class Task(Action):
    """Action whose type should be determined by the AI."""
    prompt: str

class TaskList(Action):
    """List of tasks to be done by the AI."""
    tasks: list[Task]


# tasks and actions database
tasksToDo:   list[Task]   = [] # this is updated in PushTasks() and PopTask()
tasksDone:   list[Task]   = [] # this is updated in BotSaveSummary.Use()
actionsDone: list[Action] = [] # this is updated in Bot.Loop()


def PushTasks(taskList: TaskList) -> None:
    """Adds tasks to the to-do list."""
    # we add taks in the beginning, because they are subtasks
    # Before: A, B, C, D, E
    # After: A1, A2, A3, B, C, D, E
    for task in reversed(taskList["tasks"]):
        tasksToDo.insert(0, task)


def PopTask() -> t.Optional[Task]:
    """Removes the first task from the to-do list."""
    return tasksToDo.pop(0) if len(tasksToDo) > 0 else None


class BotSaveTasks(Skill[TaskList, Task]):
    """Saves all the tasks to be done by the AI."""

    def UseWhen(self) -> str:
        return "the task to perform is complex and it can be divided into subtasks"

    def InputExample(self) -> OneOrList[TaskList]:
        return {
            "tasks": [
                {"code": "A5.T1", "prompt": "Create a new folder"},
                {"code": "A5.T2", "prompt": "Initialize a git repository"},
                {"code": "A5.T3", "prompt": "Create a README.md file"},
                {"code": "A5.T4", "prompt": "Commit the changes"},
                {"code": "A5.T5", "prompt": "Push the changes to the remote repository"},
            ]}

    def Use(self, input_: TaskList) -> Task:
        """Saves all the tasks, then starts with the first one."""
        PushTasks(input_)
        return PopTask()

    def FollowUpMessage(self, output: Task) -> str:
        return f"Now let's focus on task '{output['code']}': '{output['prompt']}'."

    def FollowUpActions(self) -> list[str]:
        return actionsAfterTask


# =============================================================================
# Summarizing a task result


# this action is used to close a task and focus on the next one
class Summary(Action):
    """Summary of a task, written by the AI."""
    summary: str


class BotSaveSummary(Skill[Summary, Task]):
    """Saves the summary of a task."""

    def UseWhen(self) -> str:
        return "the task is finished and we want to summarize the result"

    def InputExample(self) -> OneOrList[Summary]:
        return [
            {"code": "A5.S",
                "summary": "We created a new folder, initialized a git repository,"
                    + "created a file and pushed the changes to the remote repository on GitHub."
            },
            {"code": "A7.S",
                "summary": "We prepared a detailed roadmap to achieve the main goal,"
                 " starting from writing the project requirements in a markdown file,"
                + " ending with preparing a marketing campaign for the product."
            },
        ]

    def Use(self, input_: Summary) -> Task:
        tasksDone.append(input_)
        return PopTask() or {"code": "END", "prompt": "No more tasks to do!"}

    def FollowUpMessage(self, output: Task) -> str:
        if output["code"] == "END": return ""
        return f"Now let's focus on task '{output['code']}':\n'{output['prompt']}'."

    def FollowUpActions(self) -> list[str]:
        return actionsAfterTask


# =============================================================================
# Asking the user for information


class UserQuestion(Action):
    """Question to be answered by the user."""
    question: str
    suggestions: list[str]

class UserQuestions(Action):
    """Questions to be answered by the user."""
    questions: list[UserQuestion]

class UserAnswer(UserQuestion):
    """Answer by the user to a question."""
    answer: str

class UserAnswers(Action):
    """Answers by the user to questions."""
    answers: list[str]


class BotAskUser(Skill[UserQuestions, UserAnswers]):
    """Asks the user for information."""

    def UseWhen(self) -> str:
        return "the task requires some information from the user." \
            + " Try this only if you are stuck."

    def InputExample(self) -> OneOrList[UserQuestions]:
        return {"questions": [
            {"code": "A5.Q1",
                "question": "Which language should we use to write our software?",
                "suggestions": [...]
            },
            {"code": "A5.Q2",
                "question": "Which framework should we use to write our software?",
                "suggestions": [...]
            },
            {"code": "A5.Q3",
                "question": "Which database should we use to store our data?",
                "suggestions": [...]
            },
            {"code": "A5.Q4",
                "question": "Which cloud provider should we use to host our software?",
                "suggestions": [...]
            },
        ]}


    def Use(self, input_: UserQuestions) -> UserAnswers:
        """Asks the user for each question."""

        # for each question
        answers = []
        for question in input_["questions"]:

            # shows the question and the suggestions
            fullQuestion = question["question"]
            if len(question["suggestions"]) > 0:
                fullQuestion += "\n" + "\n".join([
                    "Suggestions:", *(
                        f"{s.ascii_lowercase[itemIndex]} {item}"
                        for itemIndex, item in enumerate(question["suggestions"])
                    )
                ])

            # asks for input
            answer = user.Ask(question = fullQuestion)

            # tries to recognize the suggestion index
            try:
                index = s.ascii_lowercase.index(answer.strip().lower())
                answer = question["suggestions"][index]
            except (ValueError, IndexError):
                pass

            # saves the answer
            # TODO: maybe it would be better to let the AI ask the questions
            # so that it can react promptly to strange answers
            answers.append({**question.copy(), "answer": answer})

        return {"answers": answers}


    def FollowUpMessage(self, output: UserAnswers) -> str:
        return "The user gave the following answers" \
            + " to these questions:\n" + json.dumps(output) + "\n" \
            + "Let's proceed now and put this into practice."

    def FollowUpActions(self) -> list[str]:
        return actionsAfterTask


# =============================================================================
# Searching online


class SearchQuery(Action):
    """Query to be searched online by the bot."""
    query: str

class SearchResult(SearchQuery):
    """Answer of a query searched online by the bot."""
    result: str


class BotSearchOnline(Skill[SearchQuery, SearchResult]):
    """Searches online for the given query."""

    def UseWhen(self) -> str:
        return "the task requires some information from the internet"

    def InputExample(self) -> OneOrList[SearchQuery]:
        return [
            {"code": "A5.Q1", "query": "glpk download direct"},
            {"code": "A7.Q1", "query": "how to install llama2 on windows"},
        ]

    def Use(self, input_: SearchQuery) -> SearchResult:
        # TODO: uses LangChain DuckDuckGo agent to search online
        return {"result": "https://duckduckgo.com/?q=" + input_["query"]}

    def FollowUpMessage(self, output: SearchResult) -> str:
        return "Here is the result of your search:\n" + output['result'] + "\n" \
            + "Let's proceed now and put this into practice."

    def FollowUpActions(self) -> list[str]:
        return ["Summary"]


# =============================================================================
# Choosing between options


class Option(Action):
    """Option to be selected by the AI."""
    option: str
    pros: list[str]
    cons: list[str]

class Options(Action):
    """Options the AI has to choose between."""
    options: list[str]


class BotAskOptions(Skill[Options, Options]):
    """Asks the AI to choose between options."""

    def UseWhen(self) -> str:
        return "the task can be performed in different ways," \
            + " that should be evaluated and compared"

    def InputExample(self) -> OneOrList[Options]:
        return {
            "code": "A5.O1",
            "options": [
                {"code": "A5.O1A",
                    "option": "Awesome commercial software",
                    "pros": ["it's easy to use", "very fast"],
                    "cons": ["it's not free", "cannot be extended"],
                },
                {"code": "A5.O1B",
                    "option": "Open source library",
                    "pros": ["it's free", "can be extended to our needs"],
                    "cons": ["it's not easy to use", "a bit slow"],
                },
            ]
        }

    def Use(self, input_: Options) -> Options:
        return input_

    def FollowUpMessage(self, output: Options) -> str:
        return "Which option is the best for our case?"

    def FollowUpActions(self) -> list[str]:
        return ["Choice", "Search", "UserQuestions"]


# =============================================================================
# Saving the a choice made by the AI


class Choice(Action):
    """Choice made by the AI, between options."""
    choice: str
    reasons: list[str]


class BotSaveChoice(Skill[Choice, Choice]):
    """Saves the choice made by the AI."""

    def UseWhen(self) -> str:
        return "the task requires the AI to choose between different options"

    def InputExample(self) -> OneOrList[Choice]:
        return {
            "code": "A5.C1",
            "choice": "Awesome commercial software",
            "reasons": [
                "we really need somwthing fast",
                "we don't need so much customization"
            ],
        }

    def Use(self, input_: Choice) -> Choice:
        return input_

    def FollowUpMessage(self, output: Task) -> str:
        return "Ok let's remember our choice!"

    def FollowUpActions(self) -> list[str]:
        return ["UserQuestions", "SearchQuery", "Options"]


# =============================================================================
# Running shell commands


class Command(Action):
    """Shell command to be executed by the bot."""
    command: str
    timeout: int

class CommandResult(Command):
    """Output of a shell command."""
    output: str
    elapsedTime: int


class BotRunCommand(Skill[Command, CommandResult]):
    """Runs a shell command."""

    def UseWhen(self) -> str:
        return "the task requires running a shell command." \
            "This is needed to perform every practical task."

    def InputExample(self) -> OneOrList[Command]:
        return [
        {"code": "A5.R1", "command": "ls -l",        "timeout": 5},
        {"code": "A7.R1", "command": "git status",   "timeout": 5},
    ]

    def Use(self, input_: Command) -> CommandResult:
        """Runs the given shell command."""

        # dangerous commands that require confirmation
        shellConfirm = True
        dangerousCmds = ["rm", "del", "reboot", "shutdown"]

        # if the command is dangerous, if the option is enabled
        if any(cmd in input_["command"] for cmd in dangerousCmds) and shellConfirm:

            # asks the user for confirmation
            response = user.Ask("Are you sure you want to" \
                + f" run the command '{input_['command']}'? [Y/n]"
            )
            if response not in ["y", "yes", ""]:
                return {"output": "Command cancelled by the user.", "elapsedTime": 0}

        # starts the timer and runs the command
        time = tm.time()
        try:
            output = sp.check_output(input_["command"],
                stderr = sp.STDOUT, timeout = input_["timeout"])
        except sp.CalledProcessError as e:
            output = e.output

        return {"output": output, "elapsedTime": tm.time() - time }

    def FollowUpMessage(self, output: CommandResult) -> str:
        """Shows the command output and the elapsed time."""
        return f"Elapsed time: {output['elapsedTime']} seconds.\n" \
            + f"Command output: \n{output['output']}"

    def FollowUpActions(self) -> list[str]:
        return actionsAfterResult


# =============================================================================
# Actors and user interface


@dc.dataclass
class User:
    name: str = "User"

    def Ask(self, question: str) -> str:
        """Asks the user for input."""
        print(bot.name + ": " + question)
        response = input(f"{self.name}: ")
        return response


@dc.dataclass
class AI:
    name: str = "AI"
    memory: lc.memory.ConversationBufferMemory = None # type: ignore

    showQuestions: bool = True
    showAnswers:   bool = True

    def __post_init__(self) -> None:
        """Sets up the memory."""
        self.memory = lc.memory.ConversationBufferMemory(
            memory_key = "history", return_messages = True)

    def GetChain(self, followUpActions: list[str]) -> lc.chains.LLMChain:
        """Prepares LangChain and the system prompt."""

        # builds the actions and examples
        actionsAndExamples = "\n".join([
            f"ActionType: '{skill.inputActionType}'\n" +
            f"Use when: {skill.UseWhen()}\n" +
            skill.FormattedInputExamples()
            for skill in skills
        ])

        systemPrompt = f"""
            You are a helpful assistant who generates suggestions in JSON format.
            We are a magic trio: '{bot.name}' (me), {self.name} (you) and the human '{user.name}'.
            This is our goal that we will reach: {bot.goal}

            We will cooperate to achieve this goal, and we will do it in the following way:\n
                1. I will give you a task to do
                2. You will respond choosing the best action to perform to reach our goal
                3. I will perform the action, and I will tell you the result
                4. I will provide you a list of possible next actions
                5. We will repeat this until our goal will be reached

            Here the actions you can choose, with some examples:

            {actionsAndExamples}

            You can ONLY respond with JSON objects, otherwise I will not understand you.
            You must always include the 'actionType' field, to specify the type of action.
            You must always include the 'code' field, and it must be unique for each action.
            For task codes start with A, B, C; then A1, A2, A3 for subtasks of A; then A1.A, A1.B, A1.C, etc.
            You must always include the other fields shown in the examples, according to the action type.
            You must not include other fields, otherwise I will not understand you.
        """

        # escapes the curly brackets, otherwise they are interpreted as placeholders
        systemPrompt = systemPrompt.replace("{", "{{").replace("}", "}}")

        # follow-up message for the AI
        followUpPrompt = f"""
            Now please respond with the JSON of one of the following action types:
            {', '. join(followUpActions)}.
            Include "code" and "actionType" fields, plus all the other ones shown in the examples.
            Include only the JSON of the action, nothing else.
        """

        # builds the prompt
        prompt = lc.prompts.chat.ChatPromptTemplate(messages=[
            lc.prompts.chat.SystemMessagePromptTemplate.from_template(systemPrompt),
            lc.prompts.chat.MessagesPlaceholder(variable_name = "history"),
            lc.prompts.chat.HumanMessagePromptTemplate.from_template("{input}"),
            lc.prompts.chat.HumanMessagePromptTemplate.from_template(followUpPrompt),
        ])

        #prompt = lc.prompts.PromptTemplate(
        #    template = "\n\n".join([systemPrompt, "{history}", "{input}", followUpPrompt]),
        #    input_variables = ["history", "input"],
        #)

        # sets up the model with memory
        return lc.chains.LLMChain(llm = lc.llms.OpenAI(),
            prompt = prompt, memory = self.memory)


    def Ask(self, question: str, followUpActions: list[str]) -> str:
        """Asks the AI for input."""

        # if configured, shows the question
        if self.showQuestions:
            print(bot.name + ": " + question)

        # builds the chain and gets the AI response
        response = self.GetChain(followUpActions).run({"input": question})
        
        # removes "AI: " before the start of the line
        response = response.strip("\nAI: ")

        # if configured, shows the response
        if self.showAnswers:
            print(self.name + ": " + response)

        return response


@dc.dataclass
class Bot:
    name: str = "Bot"

    # main goal, task to be done
    goal: str = ""

    def Welcome(self) -> None:
        """Welcomes the user and asks for the main goal."""

        # no need to welcome the user if the goal is already set
        if self.goal != "": return

        # asks for the main goal, until the user provides one
        while self.goal == "":
            self.goal = user.Ask("Hello! What is your main goal?")

        # confirms the main goal
        print(f"{self.name}: Great! Me and the AI will help you with it.")
        print(f"{ai.name}: Yes! Let's create some magic!")


    # stops the loop at regular intervals
    stopEvery: int = 5

    def Loop(self) -> None:
        """Keep asking the AI for actions to perform."""

        # follow-up message for the AI
        followUpMessage = "Let's start!"
        followUpActions = actionsAfterTask

        # loop
        while True:
            for _ in range(self.stopEvery):

                # asks the AI for the next action
                jsonResponse = ai.Ask(followUpMessage, followUpActions)
                action = json.loads(jsonResponse)

                # gets the skill, and uses it
                skill = GetSkill(action)
                output = skill.Use(action)
                actionsDone.append(action)

                # stops the loop if the task is finished
                if output.get("code", "") == "END": return

                # updates the follow-up message and actions
                followUpMessage = skill.FollowUpMessage(output)
                followUpActions = skill.FollowUpActions()

            # stops for a while
            input("Press Enter to continue...")


    def Celebrate(self) -> None:
        """The bot shows his happiness for completing the task."""

        print(f"{self.name}: Great! We completed the task!")
        print(f"{ai.name}: What a magic trio! We are the best!")


# =============================================================================
# Program entry point


if __name__ == "__main__":

    # sets up the actors
    user = User()
    bot  = Bot()
    ai   = AI()

    # gets the main goal from the command line
    bot.goal = sys.argv[1] if len(sys.argv) > 1 else ""

    # if needed welcomes the user and asks for the main goal
    bot.Welcome()

    # starts the loop
    bot.Loop()

    # celebrates the completion of the task
    bot.Celebrate()
