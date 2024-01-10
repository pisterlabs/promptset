from typing import Literal
from attr import NOTHING

from exports import export
import jinja2

# from templated.templates.__old_template import Template
from templated.templates.base_template import BaseTemplate as Template
from templated._utils.llms import LLM_CONSTRUCTORS, make_default_llm


@export
class Function:
    """
    Simple example:
        >>> from templated import Function
        >>> f = Function("Hello, {{name}}!")
        >>> result = f(name="world")
        >>> type(print(result)) == str
        True

    Stateful template args:
        >>> f2 = Function("{{greeting}}, {{name}}!")
        >>> f2(greeting="Hello")
        >>> f2(name="world")
        Hello, world!
        >>> f2(greeting="Goodbye")
        Goodbye, world!
        >>> f2()
        Goodbye, world!

    Custom LM:
        >>> from langchain.chat_models import ChatOpenAI
        >>> from templated._utils.chat2vanilla_lm import Chat2VanillaLM
        >>> from templated.function import Function
        >>> LLMChatOpenAI = Chat2VanillaLM(ChatOpenAI)
        >>> f = Function("Hello, {{name}}!", llm=LLMChatOpenAI)
        >>> assert f._template == "Hello, {{name}}!"
        >>> result = f(name="world")
        >>> type(print(result)) == str
        True
    """

    def __init__(
        self,
        template: str | jinja2.Template | Template,
        format: Template.Format = None,
        llm=None,
        verbose=False,
        **llm_kwargs,
    ):
        """
        Initializes a Parsel template (str | jinja2.Template | Template): The template to use for the function.
        format (Template.Format, optional): The format of the template. Inferred from the template by default.
        llm (callable, optional): The language model to use for the function. Defaults to templated._utils._DEFAULT_LLM.
        verbose (bool, optional): Whether to print verbose output. Defaults to False.
        **llm_kwargs: Additional arguments to pass to the language model constructor.
        """
        self.template = Template.create_template(template, format=format)
        self.vars = {k: NOTHING for k in self.template.vars}
        self.llm = llm or make_default_llm(llm_kwargs=llm_kwargs)
        self.verbose = verbose

    def __call__(self, **kwargs) -> str | None:
        """
        Calls the function with the specified keyword arguments.

        Args:
            **kwargs: The keyword arguments to pass to the function.

        Returns:
            str | None: If all template variables are filled, returns the response to the template. Otherwise, returns self.
        """
        self.vars.update(kwargs)
        if NOTHING in self.vars.values():
            return self
        rendered_template = self.template.render(**self.vars)
        if self.verbose:
            print(f"Rendered template: {rendered_template}")
        return self.llm(rendered_template)
