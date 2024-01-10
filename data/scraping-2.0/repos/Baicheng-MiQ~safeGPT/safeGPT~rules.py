"""
This file contains the rules for the safety system.
All classes must have a check method that returns a boolean.
You can add your own rules by creating a new class that inherits from Rule.
"""

import openai
import safeGPT
import inspect
from typing import Optional, List, Callable
from abc import ABC, abstractmethod


class Rule(ABC):
    """All rules must inherit from this class"""
    @abstractmethod
    def check(self, input_text: str) -> bool:
        """
        This method checks if the input text is safe or not.

        :param input_text: The text to check
        :return: True if the text is safe, False otherwise
        """
        return False

class OpenAIModeration(Rule):
    """This rule uses OpenAI's moderation API to check for bad content."""
    __slots__ = "categories"

    def __init__(self, categories: Optional[List[str]] =None):
        """
        :param categories: A list of categories to check for. If None, all categories will be checked.
        """
        possible_categories = [
            "hate",
            "hate/threatening",
            "self-harm",
            "sexual",
            "sexual/minors",
            "violence",
            "violence/graphic"
        ]
        if categories is None:
            self.categories: [str] = possible_categories
        elif len(categories) == 0:
            print("safeGPT: Warning: No categories specified. No categories will be checked.")
        else:
            if any([category not in possible_categories for category in categories]):
                raise ValueError("safeGPT: Invalid category"+
                                str([category for category in categories
                                     if category not in possible_categories]))
            self.categories: [str] = categories

    def check(self, input_text: str) -> bool:
        """
        checks if any of the categories are in the input text

        :param input_text: The text to check
        :return: True if any of the categories are detected, False otherwise
        """
        openai.api_key = safeGPT.api_key
        res = openai.Moderation.create(
            input = input_text,
        )['results'][0]['categories']
        return any(res[category] for category in self.categories)


class KeywordDetection(Rule):
    # This rule detects keywords in the input text.

    __slots__ = "keywords"
    def __init__(self, keywords: [str]):
        """
        :param keywords: A list of keywords to check for.
        """
        self.keywords: [str] = keywords

    def check(self, input_text: str) -> bool:
        """
        checks if any of the keywords are in the input text

        :param input_text: The text to check
        :return: True if any of the keywords are detected, False otherwise
        """
        return any(keyword in input_text for keyword in self.keywords)


class RegexSearch(Rule):
    # This rule matches the input text against a regex.

    __slots__ = "regex"
    def __init__(self, regex: str):
        import re
        """
        :param regex: The regex to check for.
        """
        # check invalid regex
        try:
            re.compile(regex)
        except re.error:
            raise ValueError("Invalid regex")

        self.regex: str = regex

    def check(self, input_text: str) -> bool:
        """
        checks if the regex matches the input text

        :param input_text: The text to check
        :return: True if the regex matches, False otherwise
        """
        import re
        return re.search(self.regex, input_text) is not None


class DoNotFlag(Rule):
    # This rule always returns false.
    def check(self, input_text: str) -> bool:
        """
        always returns false

        :param input_text: The text to check
        :return: False
        """
        return False

class AlwaysFlag(Rule):
    # This rule always returns true.
    def check(self, input_text: str) -> bool:
        """
        always returns true

        :param input_text: The text to check
        :return: True
        """
        return True

# custom rules (a decorator)
def custom_rule(func: Callable[[str], bool]) -> Rule:
    """
    This decorator can be used to create custom rules.
    The function must take a string as input and return a boolean.

    Example:

        @custom_rule\n
        def my_rule(input_text: str) -> bool:
            return "keyword" in input_text
    """
    class CustomRule(Rule):
        def __init__(self, input_func: Callable[[str], bool]):
            # check if the function is valid
            if not inspect.isfunction(input_func):
                raise ValueError("safeGPT: Custom rule must be a function")
            self.func = input_func

        def check(self, input_text: str) -> bool:
            try:
                res = self.func(input_text)
            except Exception as e:
                raise ValueError("safeGPT: Custom rule raised an error: "+str(e)+
                                 "\n\nA likely cause is your function is not type Callable[[str], bool]")
            if not isinstance(res, bool):
                raise ValueError("safeGPT: Custom rule must return a boolean, got "+str(type(res)))
            return res
    return CustomRule(func)

class SequentialCheck(Rule):
    """
    This rule checks if all the rules in the list return true.
    Useful if you want to apply multiple rules.
    """
    __slots__ = "rules"
    def __init__(self, rules: [Rule]):
        """
        :param rules: A list of rules to check.
        A good practice is to put the most expensive rules last in the list, this way the most expensive rules will only be checked if the cheaper rules return false.
        """
        self.rules: [Rule] = rules
        # check if all rules are valid
        for rule in self.rules:
            if not isinstance(rule, Rule):
                raise ValueError("safeGPT: Invalid rule: "+str(rule))

    def __len__(self):
        return len(self.rules)

    def check(self, input_text: str) -> bool:
        """
        checks if any of the rules return true

        :param input_text: The text to check
        :return: True if any of the rules return true, False otherwise
        """
        return any(rule.check(input_text) for rule in self.rules)