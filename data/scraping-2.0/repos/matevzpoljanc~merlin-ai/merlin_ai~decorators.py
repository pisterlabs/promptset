"""
Decorators for AI classes
"""
from typing import Type, Optional, Callable, Union

from merlin_ai.ai_classes import OpenAIModel, BaseAIClass, OpenAIEnum


def decorator_generator(ai_class: Type[BaseAIClass]):
    """
    Generator for decorators
    :param ai_class: AI Class type for which to  generate a decorator
    """

    def ai_class_decorator(*args, **kwargs) -> Union[BaseAIClass, Callable]:
        """
        Turn base class into AI Model
        """

        def _create_ai_model(model_settings: Optional[dict] = None):
            """
            Create AI Model
            :param model_settings:
            :return:
            """

            def _inner(base_class: Type):
                """
                Inner function
                :param base_class:
                :return:
                """
                return ai_class(base_class, model_settings=model_settings)

            return _inner

        if len(args) == 1 and callable(args[0]):
            return _create_ai_model(kwargs)(args[0])

        return _create_ai_model(kwargs)

    return ai_class_decorator


ai_model = decorator_generator(OpenAIModel)
ai_enum = decorator_generator(OpenAIEnum)
