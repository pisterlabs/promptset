import os

import openai 
import textwrap
import loguru
from IPython.display import display, Markdown
from abc import ABC, abstractmethod

from .connectors import (
    GitHubConnector,
)
from .util import (
    spinning
)



class MessageChannel(ABC):
    """Abstract base class for message sinks."""

    @abstractmethod
    def post(self, message):
        """Post a message to the sink.

        Args:
            message (str): The message to post.
        """
        pass

class ToPrint(MessageChannel):
    """Prints messages to the console."""

    def post(self, message, issue=None):
        """Post a message to the console.

        Args:
            message (str): The message to post.
        """
        print()
        print()
        wrapped_text = textwrap.fill(message, width=80)
        print(wrapped_text)

class ToDisplay(MessageChannel):
    """Displays messages as Markdown in a Jupyter notebook."""

    def post(self, message, issue=None):
        """Post a message as Markdown in a Jupyter notebook.

        Args:
            message (str): The message to post.
        """
        display(Markdown(message))

class ToGithubIssue(MessageChannel):
    """Posts messages to a GitHub repository."""

    def __init__(self, issue):
        self.issue = issue

    def post(self, message):
        """Post a message to the GitHub repository.

        Args:
            message (str): The message to post.
        """
        self.issue.create_comment(message)

class Bot:
    pass

class TrIAge(Bot):
    """ A helpful bot assisting users and maintainers of open source projects. """

    name = "trIAge"
    mission = f"You are {name}, a helpful bot that assists users and maintainers of open source projects."

    jobs = [
        "You are a bot that helps users and maintainers of open source projects.",
        "You are able to assess and rate the quality of issues.",
        "You are able to give suggestions on how to improve the quality of issues.",
        "You are able to point users to relevant documentation and other ressources.",
        "You are able to suggest solutions to issues.",
    ]

    instructions = [
        "When addresssing a user, you should always @-mention their username.",
        "When asked to respond, you should only output the response, not acknowledge the request.",
        "Your answers should be polite but concise and to the point.",
    ]


    def __init__(
        self, 
        model_provider,
        model_api_key,
        hub_api_key, 
        model_name="gpt-3.5-turbo",
        channel=ToPrint(),
    ):
        self.model_provider = model_provider
        self.model_name = model_name
        self.set_api_key(model_api_key)
        self.hub = GitHubConnector(hub_api_key)
        self.channel = channel
        self._configure()


    def set_api_key(self, api_key):
        """ Set the API key for the model provider."""
        if self.model_provider == "openai":
            self.api_key = api_key
            openai.api_key = api_key
        else:
            raise NotImplementedError(f"unknown model provider {self.model_provider}")
        
    @spinning(text="Configuring...")
    def _configure(self):
        """ Configure the bot by prompting it to describe its mission and jobs. """
        self.chat_history = []
        self.chat_history.append(
            {
                "role": "system",
                "content": self.mission,
            }
        )
        for job in self.jobs:
            self.chat_history.append(
                {
                    "role": "system",
                    "content": job,
                }
            )
        for instruction in self.instructions:
            self.chat_history.append(
                {
                    "role": "system",
                    "content": instruction,
                }
            )
        openai.ChatCompletion.create(
            model=self.model_name,
            messages=self.chat_history
        )

    def post(self, message):
        """Post a message to the sink.

        Args:
            message (str): The message to post.
        """
        self.channel.post(message)

    def display(self, text):
        display(Markdown(text))
    
    def print(self, text):
        print()
        print()
        wrapped_text = textwrap.fill(text, width=80)
        print(wrapped_text)

    @spinning(text="Thinking...")
    def tell_system(
        self,
        prompt,
    ):
        self.chat_history.append({"role": "system", "content": prompt})
        response = openai.ChatCompletion.create(
            model=self.model_name,
            messages=self.chat_history
        )
        # for debugging
        self.display(response.choices[0].message["content"])


    @spinning(text="Thinking...")
    def tell(
        self,
        prompt,
    ):
        """Tell the bot something and display the response."""
        self.chat_history.append({"role": "user", "content": prompt})
        response = openai.ChatCompletion.create(
            model=self.model_name,
            messages=self.chat_history
        )
        message = response.choices[0].message
        self.chat_history.append(message)
        self.display(message["content"])
        return message

    def see_repo(self, repo_url):
        """ """
        self.hub.connect_repo(repo_url)
        repo = self.hub.repo
        # tell the bot about the repository
        facts = [
            f"You are looking at the repository {repo_url}.",
            f"The repository is called {repo.name}.",
            f"The repository description is {repo.description}.",
            f"The repository has {repo.stargazers_count} stars.",
            f"The repository is licensed under {repo.get_license().license.name}.",
        ]
        self.tell_system(" ".join(facts))
        content = [
            #f"This is the README of the repository {repo.name}:\n {self.hub.get_readme(words=None)}",
         ]
        self.tell_system(" ".join(content))

    def get_issues(self):
        """ """
        if self.hub.repo is None:
            print("Please connect to a repository first")
            return
        issues = self.hub.repo.get_issues(state="all")
        return list(issues)

    def see_issue(self, issue):
        """ """

        # tell the bot about the issue
        facts = [
            f"You are looking at the issue {issue.title}.",
            f"The issue is labeled {issue.labels}.",
            f"The issue has been filed by the user {issue.user.login}.",
            f"The issue description is {issue.body}.",
            #f"The issue has {len(issue.comments)} comments.",
            #f"The issue has been opened {issue.created_at}.",
            #f"The issue has been updated {issue.updated_at}.",
        ]
        self.tell_system(" ".join(facts))
    
    def discuss_issue(self, issue):
        """ """
        self.channel = ToGithubIssue(issue)

    def rate_quality(self, issue):
        """ """
        self.tell_system("When asked for a rating you will answer with nothing but the rating as a number, no additional text. Ratings are whole numbers.")
        message = self.tell("Rate the quality of this issue in terms of descriptiveness and reproducibility on a scale from 0 to 10.")
        rating = int(message["content"])
        return rating
    
    def rate_novelty(self, issue):
        """ """
        message = self.tell(
            """Rate the novelty of this issue on a scale from 0 to 10, where a higher number means more novel.
            The novelty is the extent to which this issue is already answered or discussed in the documentation, where 10 means that there is no relationship between the issue and the documentation and 0 means that the issue is literally answered in the documentation. 
            Output only the rating as a number."""
        )
        rating = int(message["content"])
        return rating




    
